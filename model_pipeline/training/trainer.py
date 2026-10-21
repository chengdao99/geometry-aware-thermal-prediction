"""Run a compact MAIN training epoch. Inputs: Model, sampled batches, optimizer, and device. Outputs: Average loss and real optimizer-step count."""

from __future__ import annotations

import torch
from .normalization import move_batch_to_device


def run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
    amp: bool = True,
) -> dict:
    """Run genuine forward/validation passes and optional clipped backward optimizer steps."""
    training = optimizer is not None
    model.train(training)
    loss_sum, count, optimizer_steps, skipped_steps = 0.0, 0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, enabled=training and device.type == "cuda" and amp
            ):
                output = model(batch)
                loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(
                    "The model produced a non-finite training or validation loss."
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    previous_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < previous_scale:
                        skipped_steps += 1
                    else:
                        optimizer_steps += 1
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip, error_if_nonfinite=True
                        )
                    optimizer.step()
                    optimizer_steps += 1
            loss_sum += float(loss.detach())
            count += 1
            if max_steps is not None and count >= max_steps:
                break
    if not count:
        raise ValueError("The DataLoader produced no batches.")
    return {
        "loss": loss_sum / count,
        "batches": count,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }
