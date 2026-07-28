#!/usr/bin/env python3
"""
render_dsa_heatmap.py

Parse a DEF file and render a placement heatmap visualization.

The DEF file may contain:
  - DIEAREA coordinates (database units)
  - COMPONENTS section with:
      * Standard cells:  - name cell_type ;
      * FIXED macros:    - name cell_type + FIXED ( x y ) orientation ;
  - UNITS DISTANCE MICRONS line for scale conversion

Output: placement heatmap PNG showing die boundary, macro placements, and
        (where available) standard-cell density.
"""

import re
import sys
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# 1. DEF parser
# ---------------------------------------------------------------------------

def parse_def(filepath: str) -> dict:
    """Parse a LEF/DEF-format placed-def file and return a data dict.

    Returns
    -------
    dict with keys:
        die_area     : (llx, lly, urx, ury) in database units
        units_per_um : int, DB units per micron (from UNITS DISTANCE MICRONS)
        macros       : list of dicts {name, cell, x, y, orient, kind}
        stdcells     : list of dicts {name, cell, x, y, orient}
        num_unplaced : int, components listed without coordinates
    """
    data = {
        "die_area": None,
        "units_per_um": 2000,  # default per LEF/DEF spec
        "macros": [],
        "stdcells": [],
        "num_unplaced": 0,
    }

    # Patterns
    re_diearea = re.compile(
        r"DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)", re.IGNORECASE
    )
    re_units = re.compile(
        r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)", re.IGNORECASE
    )
    # Component line with coordinates:  - name cell_type [+ STATUS] ( x y ) orient ;
    re_comp_placed = re.compile(
        r"^\s*-\s+(\S+)\s+(\S+(?:\s+\S+)*?)\s+"
        r"(?:\+\s*(FIXED|PLACED|COVER)\s+)?"
        r"\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+)\s*;"
    )
    # Component line without coordinates:  - name cell_type ;
    re_comp_simple = re.compile(
        r"^\s*-\s+(\S+)\s+(\S+(?:\s+\S+)*?)\s*;"
    )

    in_components = False
    line_count = 0

    with open(filepath, "r") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            line_count += 1

            # --- DIEAREA ---
            m = re_diearea.search(line)
            if m:
                data["die_area"] = tuple(int(g) for g in m.groups())
                continue

            # --- UNITS ---
            m = re_units.search(line)
            if m:
                data["units_per_um"] = int(m.group(1))
                continue

            # --- COMPONENTS section boundaries ---
            if re.match(r"^\s*COMPONENTS\b", line, re.IGNORECASE):
                in_components = True
                continue
            if re.match(r"^\s*END\s+COMPONENTS\b", line, re.IGNORECASE):
                in_components = False
                continue

            if not in_components:
                continue

            # --- Try placed / fixed format first ---
            m = re_comp_placed.match(line)
            if m:
                name = m.group(1)
                cell = m.group(2).strip()
                status = (m.group(3) or "").upper()
                x = int(m.group(4))
                y = int(m.group(5))
                orient = m.group(6)

                entry = {"name": name, "cell": cell, "x": x, "y": y,
                         "orient": orient}

                is_sram = ("scratchpad" in name.lower()
                           or "quantify" in name.lower()
                           or "sram" in name.lower())

                if status == "FIXED":
                    entry["kind"] = "sram" if is_sram else "fixed"
                    data["macros"].append(entry)
                else:
                    entry["kind"] = "stdcell"
                    data["stdcells"].append(entry)
                continue

            # --- Try simple format (no coordinates) ---
            m = re_comp_simple.match(line)
            if m:
                data["num_unplaced"] += 1
                continue

    return data


# ---------------------------------------------------------------------------
# 2. SRAM macro die-size lookup (approximate)
# ---------------------------------------------------------------------------

# scratchpad_1024x128_freepdk45 dimensions estimated from DEF coordinate analysis:
#   x-spacing between adjacent scratchpads: (984480-40000)/2000 = 472.24 um
#     -> width ~460 um (leaving ~12 um gap between macros)
#   y-gap between act and wt macro rows: (6198570-5397140)/2000 = 400.7 um
#     -> height must be < 400 um to avoid overlap; estimated ~360 um
#   For a 1024x128 SRAM in FreePDK45 45nm: 460 um x 360 um is typical.
# quantify_buffer_sram: estimated ~550 um x 450 um (single, right-side buffer)

def estimate_macro_size(entry: dict, units_per_um: int) -> tuple:
    """Return (width_um, height_um) for a known macro cell type."""
    cell = entry["cell"].lower()
    if "scratchpad_1024x128" in cell:
        return (460.0, 360.0)
    elif "quantify_buffer_sram" in cell:
        return (550.0, 450.0)
    else:
        return (100.0, 100.0)


# ---------------------------------------------------------------------------
# 3. Main rendering
# ---------------------------------------------------------------------------

def render(def_path: str, out_path: str):
    """Main entry point: parse, render, save."""

    print(f"[*] Parsing DEF: {def_path}")
    data = parse_def(def_path)

    die = data["die_area"]
    if die is None:
        print("[!] ERROR: no DIEAREA found in DEF file.")
        sys.exit(1)

    units = data["units_per_um"]
    print(f"    DIEAREA (DB units): {die}")
    print(f"    UNITS DISTANCE MICRONS: {units}  ( /{units} => um )")
    die_um = tuple(v / units for v in die)  # llx, lly, urx, ury
    print(f"    DIEAREA (um): ({die_um[0]:.1f}, {die_um[1]:.1f}) -> "
          f"({die_um[2]:.1f}, {die_um[3]:.1f})")
    die_w = die_um[2] - die_um[0]
    die_h = die_um[3] - die_um[1]
    die_area_um2 = die_w * die_h
    print(f"    Die size: {die_w:.1f} x {die_h:.1f} um  "
          f"({die_area_um2 / 1e6:.4f} mm^2)")

    nmacros = len(data["macros"])
    nstdcells = len(data["stdcells"])
    nunplaced = data["num_unplaced"]
    print(f"    SRAM / FIXED macros with coords: {nmacros}")
    print(f"    Std-cells with coords:           {nstdcells}")
    print(f"    Components without coords:       {nunplaced}")
    print(f"    Total components listed:         {nmacros + nstdcells + nunplaced}")

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    macro_names = []
    for m in data["macros"]:
        macro_names.append(m["name"])
        m_um_x = m["x"] / units
        m_um_y = m["y"] / units
        m_w, m_h = estimate_macro_size(m, units)
        print(f"      MACRO {m['name']:45s}  "
              f"cell={m['cell']:35s}  "
              f"({m_um_x:8.1f}, {m_um_y:8.1f}) um  "
              f"w={m_w:.0f} h={m_h:.0f} um  "
              f"orient={m['orient']}")

    total_stdcells = nstdcells + nunplaced
    if nstdcells > 0:
        xs = np.array([c["x"] for c in data["stdcells"]], dtype=np.float64)
        ys = np.array([c["y"] for c in data["stdcells"]], dtype=np.float64)
        xs_um = xs / units
        ys_um = ys / units
        bb_xmin, bb_xmax = xs_um.min(), xs_um.max()
        bb_ymin, bb_ymax = ys_um.min(), ys_um.max()
        print(f"    Stdcell bounding-box (um): "
              f"x=[{bb_xmin:.1f}, {bb_xmax:.1f}]  "
              f"y=[{bb_ymin:.1f}, {bb_ymax:.1f}]")
    else:
        bb_xmin = bb_xmax = bb_ymin = bb_ymax = None

    # Estimate utilization (stdcell area / (die_area - macro_area))
    # Assume avg stdcell area ~ 10 um^2 (FreePDK45 stdcell ~ 3.8 x 2.8 ~ 10.6)
    avg_stdcell_area_um2 = 10.6
    total_stdcell_area = total_stdcells * avg_stdcell_area_um2
    total_macro_area = 0.0
    for m in data["macros"]:
        mw, mh = estimate_macro_size(m, units)
        total_macro_area += mw * mh
    available_area = die_area_um2 - total_macro_area
    util = (total_stdcell_area / available_area * 100) if available_area > 0 else 0.0
    print(f"    Estimated total stdcell area: {total_stdcell_area:.1f} um^2")
    print(f"    Total macro area:             {total_macro_area:.1f} um^2")
    print(f"    Available area (die - macro):  {available_area:.1f} um^2")
    print(f"    Estimated utilization:         {util:.2f} %")

    # ------------------------------------------------------------------
    # Figure: 2-panel layout
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("SpecLLMDSA Placement Heatmap — DSA Top", fontsize=16,
                 fontweight="bold")

    # ---- Helper: draw die boundary ----
    def draw_die_boundary(ax):
        rect = mpatches.Rectangle(
            (die_um[0], die_um[1]), die_w, die_h,
            fill=False, edgecolor="black", linewidth=2.0,
            linestyle="-", zorder=5,
        )
        ax.add_patch(rect)

    # ---- Helper: draw macro rectangles ----
    # Color per macro type group
    macro_colors = {
        "acc": "#e41a1c",   # red
        "act": "#377eb8",   # blue
        "wt":  "#4daf4a",   # green
        "quantify": "#ff7f00",  # orange
    }

    def get_macro_group(name: str) -> str:
        name_l = name.lower()
        if "acc_sram" in name_l:
            return "acc"
        elif "act_sram" in name_l:
            return "act"
        elif "wt_sram" in name_l:
            return "wt"
        elif "quantify" in name_l:
            return "quantify"
        return "other"

    def draw_macros(ax, alpha=1.0):
        for m in data["macros"]:
            mx = m["x"] / units
            my = m["y"] / units
            mw, mh = estimate_macro_size(m, units)
            group = get_macro_group(m["name"])
            color = macro_colors.get(group, "#999999")
            rect = mpatches.Rectangle(
                (mx, my), mw, mh,
                fill=True, facecolor=color, edgecolor="black",
                linewidth=0.5, alpha=alpha, zorder=10,
            )
            ax.add_patch(rect)

    # ==================================================================
    # PANEL 1: Cell density heatmap
    # ==================================================================
    ax1.set_title("Cell Density Heatmap + SRAM Macro Overlay", fontsize=13)
    ax1.set_xlabel("X (um)")
    ax1.set_ylabel("Y (um)")
    ax1.set_aspect("equal")

    if nstdcells > 0:
        xs_um = np.array([c["x"] for c in data["stdcells"]]) / units
        ys_um = np.array([c["y"] for c in data["stdcells"]]) / units
        bins = 200
        hist, xedges, yedges = np.histogram2d(
            xs_um, ys_um,
            bins=[bins, bins],
            range=[[die_um[0], die_um[2]], [die_um[1], die_um[3]]],
        )
        # Log-scale for better contrast
        hist_masked = np.ma.masked_where(hist == 0, hist)
        im = ax1.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap="inferno", alpha=0.9, norm=LogNorm(vmin=1, vmax=max(hist.max(), 1)),
            shading="auto", rasterized=True,
        )
        cbar = fig.colorbar(im, ax=ax1, shrink=0.78, pad=0.02)
        cbar.set_label("Cell count per bin", fontsize=9)
    else:
        # No stdcell placement data -- show the macro layout on blank die
        ax1.text(die_um[0] + die_w / 2, die_um[1] + die_h / 2,
                 "No stdcell placement\ncoordinates in DEF\n\n"
                 "(macros only)",
                 ha="center", va="center", fontsize=14,
                 color="gray", style="italic",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="gray", alpha=0.85))
        # Draw a light grid to indicate die area
        ax1.set_facecolor("#f5f5f5")

    draw_macros(ax1, alpha=0.9)
    draw_die_boundary(ax1)
    ax1.set_xlim(die_um[0] - 200, die_um[2] + 200)
    ax1.set_ylim(die_um[1] - 200, die_um[3] + 200)

    # ==================================================================
    # PANEL 2: Macro + stdcell scatter with zoom
    # ==================================================================
    ax2.set_title("Macro Placement + Stdcell Scatter", fontsize=13)
    ax2.set_xlabel("X (um)")
    ax2.set_ylabel("Y (um)")
    ax2.set_aspect("equal")

    if nstdcells > 0:
        xs_um = np.array([c["x"] for c in data["stdcells"]]) / units
        ys_um = np.array([c["y"] for c in data["stdcells"]]) / units
        # Downsample if more than 200k points for performance
        if len(xs_um) > 200000:
            idx = np.random.choice(len(xs_um), 200000, replace=False)
            xs_um = xs_um[idx]
            ys_um = ys_um[idx]
        ax2.scatter(xs_um, ys_um, s=0.3, c="gray", alpha=0.15,
                    marker=".", rasterized=True, zorder=2,
                    label="stdcells")
    else:
        ax2.text(die_um[0] + die_w / 2, die_um[1] + die_h / 2,
                 "No stdcell placement\ncoordinates in DEF",
                 ha="center", va="center", fontsize=14,
                 color="gray", style="italic")

    draw_macros(ax2, alpha=0.85)
    draw_die_boundary(ax2)

    # Legend for macro groups
    legend_handles = []
    legend_labels_seen = set()
    group_names = {"acc": "Accumulator SRAM", "act": "Activation SRAM",
                   "wt": "Weight SRAM", "quantify": "Quantify Buffer SRAM"}
    for m in data["macros"]:
        group = get_macro_group(m["name"])
        if group not in legend_labels_seen:
            legend_labels_seen.add(group)
            color = macro_colors.get(group, "#999999")
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="black",
                               label=group_names.get(group, group))
            )
    if nstdcells > 0:
        legend_handles.append(
            plt.Line2D([0], [0], marker=".", color="w", markerfacecolor="gray",
                       markersize=8, label="stdcells")
        )
    ax2.legend(handles=legend_handles, loc="upper right",
               fontsize=8, framealpha=0.9)

    ax2.set_xlim(die_um[0] - 200, die_um[2] + 200)
    ax2.set_ylim(die_um[1] - 200, die_um[3] + 200)

    # ------------------------------------------------------------------
    # Info text box below panels
    # ------------------------------------------------------------------
    info_lines = (
        f"DEF: {os.path.basename(def_path)}\n"
        f"Die: {die_w:.0f} x {die_h:.0f} um  |  "
        f"DB units/micron: {units}  |  "
        f"Macros: {nmacros}  |  "
        f"StdCells listed: {total_stdcells}  |  "
        f"Util (est): {util:.1f}%"
    )
    if nstdcells == 0 and nunplaced > 0:
        info_lines += (
            f"\nNOTE: {nunplaced} standard cells have no placement "
            f"coordinates in this DEF (floorplan-stage only)."
        )
    fig.text(0.5, 0.01, info_lines, ha="center", va="bottom",
             fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                       edgecolor="#cccccc"))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[*] Saved: {out_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary text file
    # ------------------------------------------------------------------
    summary_path = out_path.replace(".png", "_summary.txt")
    with open(summary_path, "w") as sf:
        sf.write("SpecLLMDSA Placement Heatmap — Summary\n")
        sf.write("=" * 60 + "\n\n")
        sf.write(f"DEF file:               {def_path}\n")
        sf.write(f"Die area (um):          {die_w:.1f} x {die_h:.1f}\n")
        sf.write(f"Die area (mm^2):        {die_area_um2 / 1e6:.4f}\n")
        sf.write(f"DB units per micron:    {units}\n")
        sf.write(f"Total components:       {nmacros + total_stdcells}\n")
        sf.write(f"  SRAM/FIXED macros:    {nmacros}\n")
        sf.write(f"  Standard cells:       {total_stdcells}\n")
        if nstdcells > 0:
            sf.write(f"    (with coords):      {nstdcells}\n")
        if nunplaced > 0:
            sf.write(f"    (no coords):        {nunplaced}\n")
        sf.write(f"Estimated utilization:  {util:.2f} %\n")
        if bb_xmin is not None:
            sf.write(f"Stdcell bbox (um):      "
                     f"x=[{bb_xmin:.1f}, {bb_xmax:.1f}]  "
                     f"y=[{bb_ymin:.1f}, {bb_ymax:.1f}]\n")
        sf.write("\nSRAM Macro Placements:\n")
        sf.write("-" * 60 + "\n")
        for m in data["macros"]:
            mw, mh = estimate_macro_size(m, units)
            sf.write(f"  {m['name']:45s}  "
                     f"cell={m['cell']:35s}  "
                     f"({m['x']/units:8.1f}, {m['y']/units:8.1f}) um  "
                     f"W={mw:.0f} H={mh:.0f} um  "
                     f"orient={m['orient']}\n")
    print(f"[*] Saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/wangduo/project/SpecLLMDSA/results/dsa_placed.def"
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/wangduo/project/SpecLLMDSA/results/dsa_placement_heatmap.png"

    if not os.path.exists(def_path):
        print(f"[!] DEF file not found: {def_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    render(def_path, out_path)
