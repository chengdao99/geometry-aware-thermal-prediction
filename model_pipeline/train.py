"""Train the 12D MAIN model. Inputs: YAML, explicit train/val NPZ paths, and EPW. Outputs: Effective configuration, normalization, and best/last model checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import torch
import yaml
from data_pipeline.paths import DEFAULT_EPW, resolve_path
from .networks.end2end_model import ZoneOverheatPredictor
from .training.data import prepare_data
from .training.normalization import save_norm_stats, load_norm_stats
from .training.trainer import run_epoch


def main() -> None:
    """Train with explicit pairs and save compatible MAIN tensors without experiment monitoring."""
    parser = argparse.ArgumentParser(description="Train the 12D geometry-aware MAIN model.")
    parser.add_argument("--config", default="model_pipeline/configs/base.yaml")
    parser.add_argument("--train-voxel-dir")
    parser.add_argument("--train-label-dir")
    parser.add_argument("--val-voxel-dir")
    parser.add_argument("--val-label-dir")
    parser.add_argument(
        "--epw", help="Override paths.epw in the training YAML; download the recommended EPW separately."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit each train/validation epoch for an execution smoke test.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    if config.get("voxel_features", {}).get("use_element_onehot", False):
        parser.error("The public MAIN model requires 12D voxel features.")
    config["model"]["cnn_in_dim"] = 12
    config["voxel_features"] = {"use_element_onehot": False}
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if config["training"]["epochs"] < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Epochs and max-steps must be positive.")
    paths = config["paths"]
    chosen = {
        key: resolve_path(getattr(args, flag) or paths[key])
        for key, flag in [
            ("train_labels", "train_label_dir"),
            ("train_voxels", "train_voxel_dir"),
            ("val_labels", "val_label_dir"),
            ("val_voxels", "val_voxel_dir"),
        ]
    }
    epw = resolve_path(args.epw or paths.get("epw", DEFAULT_EPW))
    output = resolve_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output directory must be empty or new.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; install the documented PyTorch/spconv environment."
        )
    device = torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, norm, counts = prepare_data(
        config,
        chosen["train_labels"],
        chosen["train_voxels"],
        chosen["val_labels"],
        chosen["val_voxels"],
        epw,
    )
    save_norm_stats(norm, output / "norm_stats.npz")
    restored_norm = load_norm_stats(output / "norm_stats.npz")
    if restored_norm.temp_std != float(norm.temp_std):
        if abs(restored_norm.temp_std - norm.temp_std) > 1e-5:
            raise ValueError("Saved normalization did not round-trip.")
    model_config = copy.deepcopy(config["model"])
    model_config["loss"] = copy.deepcopy(config["loss"])
    model_config["loss"]["temperature_scale_c"] = float(norm.temp_std)
    model = ZoneOverheatPredictor(model_config).to(device)
    train = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        fused=device.type == "cuda",
    )
    epochs = int(train["epochs"])
    warmup = min(int(train["warmup_epochs"]), max(0, epochs - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup),
        eta_min=float(train["lr"]) * float(train["lr_min_ratio"]),
    )
    if warmup:
        linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linear, cosine], milestones=[warmup]
        )
    else:
        scheduler = cosine
    use_amp = bool(train.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and use_amp)
    effective = copy.deepcopy(config)
    effective["paths"] = {key: str(value) for key, value in chosen.items()}
    effective["paths"]["epw"] = str(epw)
    effective["execution"] = {"max_steps": args.max_steps, "device": str(device), **counts}
    (output / "run_config.json").write_text(
        json.dumps(effective, indent=2) + "\n", encoding="utf-8"
    )
    best, patience, total_steps = float("inf"), 0, 0
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "train_loss", "validation_loss", "optimizer_steps"]
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            result = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=float(train["grad_clip"]),
                max_steps=args.max_steps,
                amp=use_amp,
            )
            validation = run_epoch(model, val_loader, device, max_steps=args.max_steps)
            if not result["optimizer_steps"]:
                raise RuntimeError(
                    "No optimizer update occurred; the smoke test or training epoch cannot be marked successful."
                )
            scheduler.step()
            total_steps += result["optimizer_steps"]
            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_config": model_config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(payload, output / "last_model.pt")
            if validation["loss"] < best:
                best, patience = validation["loss"], 0
                torch.save(payload, output / "best_model.pt")
            else:
                patience += 1
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": result["loss"],
                    "validation_loss": validation["loss"],
                    "optimizer_steps": result["optimizer_steps"],
                }
            )
            stream.flush()
            print(
                json.dumps({"epoch": epoch, "train": result, "validation": validation, **counts}),
                flush=True,
            )
            if patience >= int(train["patience"]):
                break
    saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    check = ZoneOverheatPredictor(saved["model_config"])
    check.load_state_dict(saved["model_state"], strict=True)
    print(
        json.dumps({"checkpoint_reload": "PASS", "optimizer_steps": total_steps, **counts}),
        flush=True,
    )


if __name__ == "__main__":
    main()
