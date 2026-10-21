"""Building identifiers and optional HBJSON geometry drawings.

Inputs: Generated geometry and material records.
Outputs: Stable building identifiers and optional PNG drawings."""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict

if TYPE_CHECKING:
    from .geometric_generation.retry_divide_rooms import RoomDivisionResult
    from .geometric_generation.extrude_rooms import ExtrusionResult
    from .geometric_generation.add_windows import BuildingModel
    from .material_generation.assign_materials import MaterializedBuildingModel
logger = logging.getLogger(__name__)
_SHAPE_ABBREV: dict[str, str] = {
    "rectangle": "R",
    "quadrilateral": "Q",
    "L_shape": "L",
    "T_shape": "T",
    "U_shape": "U",
    "courtyard": "C",
}
_STRUCT_ABBREV: dict[str, str] = {
    "CLT": "CLT",
    "Concrete_NormalWeight": "CNW",
    "Concrete_Lightweight": "CLW",
    "Brick": "BRK",
    "LightWood": "LLW",
}


def _compute_avg_in_shading_coverage(building: "BuildingModel") -> float:
    total_window_area = 0.0
    shaded_window_area = 0.0
    for room in building.rooms:
        for face in room.faces:
            for apt in face.apertures:
                total_window_area += apt.area
                if apt.identifier.endswith("_shaded") or (apt.coverage and apt.coverage > 0):
                    shaded_window_area += apt.area
    if total_window_area <= 1e-06:
        return 0.0
    return shaded_window_area / total_window_area


def _build_identifier(
    mat_model: "MaterializedBuildingModel",
    shape_type: str,
    seed: int | None,
    count: int,
    rng_tag=None,
) -> str:
    g = mat_model.geometry
    ms = mat_model.material_spec
    ctx = ms.metadata.get("context", {})
    prg = ms.metadata.get("program", {})
    shape_code = _SHAPE_ABBREV.get(shape_type, shape_type[0].upper())
    struct_code = _STRUCT_ABBREV.get(ms.structural_system, ms.structural_system[:3].upper())
    area_str = f"A{int(round(g.total_floor_area)):04d}"
    height_str = f"H{int(round(g.floor_height * 100)):03d}"
    grid_str = f"G{int(round(g.grid_spacing * 100)):03d}"
    svf_raw = float(ctx.get("svf_wall", 1.0))
    svf_str = f"SVF{int(round(svf_raw * 100)):03d}"
    avg_in_shade = _compute_avg_in_shading_coverage(g)
    ishade_val = math.floor(avg_in_shade * 10)
    ishade_str = f"INS{ishade_val}"
    room_ach_day = prg.get("room_infiltration_ach_day", [])
    if room_ach_day:
        avg_ach_day = sum(room_ach_day) / len(room_ach_day)
        ach_str = f"ACH{int(round(avg_ach_day * 10)):02d}"
    else:
        ach_str = "ACH00"
    _SCENARIO_CODE = {"passive": "SCp", "occupied_passive": "SCo", "occupied_hvac": "SCh"}
    scenario_str = _SCENARIO_CODE.get(prg.get("scenario_type", "passive"), "SCp")
    count_str = f"{count:04d}"
    n_floors = mat_model.material_spec.metadata.get("building", {}).get("n_floors", 1)
    btype = mat_model.material_spec.metadata.get("building", {}).get("building_type", "real")
    floor_str = f"{('TM' if btype == 'tall_mid' else str(n_floors))}F"
    if seed is not None:
        suffix = f"S{seed}"
    else:
        suffix = f"Srand_{rng_tag}"
    return f"{shape_code}_{struct_code}_{floor_str}_{area_str}_{height_str}_{grid_str}_{ishade_str}_{svf_str}_{ach_str}_{scenario_str}_{suffix}"


def _save_plot(
    div_result: "RoomDivisionResult",
    ext_result: "ExtrusionResult",
    mat_model: "MaterializedBuildingModel",
    output_path: Path,
    identifier: str,
    seed: int | None,
    n_retries: int | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("[plot] matplotlib not installed — skipping plot generation.")
        return
    g = mat_model.geometry
    ms = mat_model.material_spec
    ctx = ms.metadata.get("context", {})
    prg = ms.metadata.get("program", {})
    u = ms.u_values
    bld = ms.metadata.get("building", {})
    fig = plt.figure(figsize=(22, 8), dpi=150)
    gs = fig.add_gridspec(
        1, 4, width_ratios=[2, 4, 2, 2], left=0.04, right=0.98, top=0.87, bottom=0.08, wspace=0.25
    )
    ax_plan = fig.add_subplot(gs[0])
    ax_iso = fig.add_subplot(gs[1], projection="3d")
    ax_info_left = fig.add_subplot(gs[2])
    ax_info_right = fig.add_subplot(gs[3])
    fig.suptitle(identifier, fontsize=10, fontweight="bold", y=0.96, family="monospace")
    footprint = div_result.footprint
    min_x, min_y, max_x, max_y = footprint.bounds
    span = max(max_x - min_x, max_y - min_y)
    margin = max(span * 0.12, 1.0)
    ax_plan.set_xlim(min_x - margin, max_x + margin)
    ax_plan.set_ylim(min_y - margin, max_y + margin)
    ax_plan.set_aspect("equal")
    ax_plan.grid(True, linestyle="--", lw=0.3, alpha=0.4)
    ax_plan.set_xlabel("m", fontsize=7)
    ax_plan.set_ylabel("m", fontsize=7)
    ax_plan.tick_params(labelsize=6)
    ext_coords = np.array(footprint.exterior.coords)
    ax_plan.plot(ext_coords[:, 0], ext_coords[:, 1], color="#2C3E50", lw=2.0, zorder=5)
    for interior in footprint.interiors:
        ic = np.array(interior.coords)
        ax_plan.fill(ic[:, 0], ic[:, 1], color="white", zorder=4)
        ax_plan.plot(ic[:, 0], ic[:, 1], color="#2C3E50", lw=1.5, zorder=5)
    for room_poly, ztype, room_id in zip(
        div_result.rooms, div_result.zone_types, div_result.room_ids
    ):
        coords = np.array(room_poly.exterior.coords)
        ax_plan.plot(coords[:, 0], coords[:, 1], color="#85929E", lw=0.7, zorder=3)
        cx, cy = (room_poly.centroid.x, room_poly.centroid.y)
        zone_label = "P" if ztype == "perimeter" else "C"
        ax_plan.text(
            cx,
            cy,
            f"{room_id}\n({room_poly.area:.1f}m²)",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#151C21",
            zorder=7,
            linespacing=1.3,
        )
    arr_cx = max_x + margin * 0.6
    arr_bot = min_y + (max_y - min_y) * 0.08
    arr_len = span * 0.11
    ax_plan.annotate(
        "",
        xy=(arr_cx, arr_bot + arr_len),
        xytext=(arr_cx, arr_bot),
        arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5, mutation_scale=10),
    )
    ax_plan.text(
        arr_cx,
        arr_bot + arr_len + span * 0.03,
        "N",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#C0392B",
    )
    ax_plan.set_title(f"Floor Plan  [rot={g.rotation_angle_deg:.1f}°]", fontsize=8.5)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    all_rooms = mat_model.geometry.rooms
    n_per_floor = ext_result.n_rooms_per_floor
    room_colors = plt.cm.Set3(np.linspace(0, 1, max(n_per_floor, 1)))
    for abs_idx, room in enumerate(all_rooms):
        color = room_colors[abs_idx % n_per_floor]
        for face in room.faces:
            verts = [face.vertices]
            if face.face_type == "Floor":
                if face.bc_type == "Surface":
                    fc = (0.55, 0.65, 0.8, 0.35)
                    ec = "#5D7A99"
                    lw = 0.5
                else:
                    fc = (0.85, 0.85, 0.85, 0.45)
                    ec = "gray"
                    lw = 0.5
            elif face.face_type == "RoofCeiling":
                if face.bc_type == "Surface":
                    fc = (0.55, 0.65, 0.8, 0.2)
                    ec = "#5D7A99"
                    lw = 0.4
                else:
                    fc = (*color[:3], 0.2)
                    ec = "gray"
                    lw = 0.5
            elif getattr(face, "is_exterior_wall", False):
                fc = (*color[:3], 0.65)
                ec = "#C0392B"
                lw = 1.2
            else:
                fc = (*color[:3], 0.3)
                ec = "#7F8C8D"
                lw = 0.35
            ax_iso.add_collection3d(
                Poly3DCollection(verts, facecolor=fc, edgecolor=ec, linewidth=lw)
            )
            for apt in face.apertures:
                ax_iso.add_collection3d(
                    Poly3DCollection(
                        [apt.vertices],
                        facecolor=(0.2, 0.5, 0.9, 0.8),
                        edgecolor="#1A5276",
                        linewidth=0.8,
                    )
                )
            for shade in face.out_shading:
                ax_iso.add_collection3d(
                    Poly3DCollection(
                        [shade.vertices],
                        facecolor=(0.95, 0.85, 0.1, 0.7),
                        edgecolor="#7D6608",
                        linewidth=0.6,
                    )
                )
    xs = [v[0] for r in all_rooms for f in r.faces for v in f.vertices]
    ys = [v[1] for r in all_rooms for f in r.faces for v in f.vertices]
    zs = [v[2] for r in all_rooms for f in r.faces for v in f.vertices]
    ax_iso.set_xlim(min(xs), max(xs))
    ax_iso.set_ylim(min(ys), max(ys))
    ax_iso.set_zlim(0, ext_result.total_height * 1.15)
    ax_iso.set_xlabel("X (m)", fontsize=7)
    ax_iso.set_ylabel("Y (m)", fontsize=7)
    ax_iso.set_zlabel("Z (m)", fontsize=7)
    ax_iso.tick_params(labelsize=6)
    ax_iso.view_init(elev=30, azim=-60)
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    z_range = ext_result.total_height * 1.15
    if x_range > 0 and y_range > 0 and (z_range > 0):
        ax_iso.set_box_aspect([x_range, y_range, z_range])
    legend_elements = [
        Patch(facecolor="#C0392B", alpha=0.7, label="Exterior wall"),
        Patch(facecolor="#7F8C8D", alpha=0.5, label="Interior wall"),
        Patch(facecolor="#BDC3C7", alpha=0.5, label="Ground/roof slab"),
        Patch(facecolor="#C1C08E", alpha=0.5, label="Inter-floor slab"),
        Patch(facecolor="#41A6FF", alpha=0.8, label="Window (clear)"),
        Patch(facecolor="#1F3950", alpha=0.8, label="Window (with shade)"),
        Patch(facecolor="#EADE75", alpha=0.7, label="Exterior shade"),
    ]
    ax_iso.legend(handles=legend_elements, fontsize=5.5, loc="upper right", ncol=2)
    ax_iso.set_title(
        f"Isometric View  [floors={ext_result.n_floors}  h={ext_result.floor_height:.1f}m]",
        fontsize=8.5,
    )
    for ax in (ax_info_left, ax_info_right):
        ax.axis("off")

    def _draw_text(ax, rows, start_y=0.98, dy=0.038, dy_gap=0.016):
        y = start_y
        for label, value in rows:
            if label == "":
                y -= dy_gap
                continue
            if label.startswith("───"):
                ax.text(
                    0.0,
                    y,
                    label,
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va="top",
                    ha="left",
                    color="#4F575F",
                    style="italic",
                )
            else:
                ax.text(
                    0.02,
                    y,
                    f"{label}:",
                    transform=ax.transAxes,
                    fontsize=7.5,
                    va="top",
                    ha="left",
                    color="#5D6D7E",
                )
                ax.text(
                    0.52,
                    y,
                    value,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    va="top",
                    ha="left",
                    color="#1C2833",
                    fontweight="bold",
                )
            y -= dy
        return y

    n_perimeter = div_result.zone_types.count("perimeter")
    n_core = div_result.zone_types.count("core")
    shade_depths = defaultdict(list)
    for room in mat_model.geometry.rooms:
        for face in room.faces:
            for shade in face.out_shading:
                orient = face.orientation
                if orient is not None:
                    shade_depths[orient].append(shade.depth)
    out_shade_str = (
        "  ".join(
            (
                f"{ori}={sum(depths) / len(depths):.2f}m"
                for ori, depths in sorted(shade_depths.items())
            )
        )
        if shade_depths
        else "none"
    )
    wwr_str = "  ".join((f"{k}={v:.2f}" for k, v in sorted(g.wwr_by_orientation.items())))
    windows_str = f"total={g.total_windows}  area={g.total_window_area:.1f} m²  wwr: {wwr_str}"
    rows_left = [
        ("─── Geometry ───", ""),
        ("Seed", str(seed) if seed is not None else "random"),
        ("Shape", g.shape_type),
        ("Area", f"{g.total_floor_area:.1f} m²"),
        ("Floor height", f"{g.floor_height:.2f} m"),
        ("Rooms", f"{g.n_rooms}"),
        ("Grid spacing", f"{div_result.grid_spacing:.2f} m"),
        ("Attempts", f"{n_retries}"),
        ("Rotation", f"{g.rotation_angle_deg:.1f}°"),
        ("Floors", f"{bld.get('n_floors', '—')} [{bld.get('building_type', '—')}]"),
        ("Inter-slab insul", str(ms.metadata.get("has_inter_slab_insulation", "—"))),
        ("Windows", f"total={g.total_windows}  area={g.total_window_area:.1f} m²"),
        ("WWR", f"{wwr_str}"),
        ("OutShades", out_shade_str),
        ("Inner shade", f"{_compute_avg_in_shading_coverage(g):.2f}"),
        ("", ""),
        ("─── BC ───", ""),
        ("Floor BC", g.floor_bc),
        ("Ceiling BC", g.ceiling_bc),
        ("", ""),
        ("─── Materials ───", ""),
        ("Structure", ms.structural_system),
        ("Insulation", ms.insulation_key),
        ("Glazing", ms.glazing_system_name),
    ]
    svf_per_floor = prg.get("svf_per_floor", [])
    if svf_per_floor:
        svf_floor_str = "  ".join(
            (f"F{s['floor_index']}={s['svf_wall']:.2f}\n" for s in svf_per_floor)
        )
    else:
        svf_floor_str = "—"
    room_ach_day = prg.get("room_infiltration_ach_day", [])
    room_ach_night = prg.get("room_infiltration_ach_night", [])
    day_start_h = prg.get("day_start_hour", 6)
    night_start_h = prg.get("night_start_hour", 22)
    if room_ach_day:
        ach_day_str = f"min={min(room_ach_day):.2f} avg={sum(room_ach_day) / len(room_ach_day):.2f} max={max(room_ach_day):.2f}"
    else:
        ach_day_str = "—"
    if room_ach_night:
        ach_night_str = f"min={min(room_ach_night):.2f} avg={sum(room_ach_night) / len(room_ach_night):.2f} max={max(room_ach_night):.2f}"
    else:
        ach_night_str = "—"

    def _feat_range(key: str) -> str:
        vals = prg.get(key, [])
        nz = [v for v in vals if abs(v) > 1e-06]
        if not nz:
            return "0 (passive)"
        return f"{min(nz):.1f}–{max(nz):.1f}  act={len(nz)}/{len(vals)}"

    equip_rng = _feat_range("room_equip_peak")
    occ_rng = _feat_range("room_occ_peak")
    hvac_modes = prg.get("room_hvac_mode", [])
    hvac_peaks = prg.get("room_hvac_peak", [])
    heat_peaks = [p for m, p in zip(hvac_modes, hvac_peaks) if m == "heating"]
    cool_peaks = [p for m, p in zip(hvac_modes, hvac_peaks) if m == "cooling"]
    if heat_peaks or cool_peaks:
        hvac_rng = f"H:{len(heat_peaks)} ({sum(heat_peaks) / len(heat_peaks):.0f} W/m²)  C:{len(cool_peaks)} ({sum(cool_peaks) / max(len(cool_peaks), 1):.0f} W/m²)"
    else:
        hvac_rng = "0 (passive/no-hvac)"
    rows_right = [
        ("─── SVF / Context ───", ""),
        ("Context scenario", ctx.get("scenario", "—")),
        ("SVF wall / roof", f"{ctx.get('svf_wall', '—')} / {ctx.get('svf_roof', '—')}"),
        ("SVF per floor", svf_floor_str),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("─── Program ───", ""),
        ("Scenario", prg.get("scenario_type", "—")),
        ("Day boundary", f"{day_start_h:02d}:00 – {night_start_h:02d}:00"),
        ("ACH day", ach_day_str),
        ("ACH night", ach_night_str),
        ("Equip [W/m²]", equip_rng),
        ("Occ [per/m²]", occ_rng),
        ("HVAC rooms", hvac_rng),
        ("", ""),
        ("─── U-values ───", ""),
        ("Wall", f"{u.get('ext_wall', float('nan')):.3f} W/m²K"),
        ("Roof", f"{u.get('roof', float('nan')):.3f} W/m²K"),
        ("Window", f"{u.get('window', float('nan')):.2f} W/m²K"),
    ]
    _draw_text(ax_info_left, rows_left, start_y=0.98, dy=0.038, dy_gap=0.016)
    _draw_text(ax_info_right, rows_right, start_y=0.98, dy=0.038, dy_gap=0.016)
    try:
        fig.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="png",
        )
        kb = output_path.stat().st_size / 1024
        logger.info(f"[plot] Saved → {output_path.name}  ({kb:.0f} KB)")
    except Exception as exc:
        logger.warning(f"[plot] Failed to save plot for {identifier}: {exc}")
    finally:
        plt.close(fig)
