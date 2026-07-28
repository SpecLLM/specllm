#!/usr/bin/env python3
# render_sram_placement.py — SRAM macro placement visualization from DEF
# Reads existing DEF, renders SRAM positions + die outline + area budget

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# ═══════════════════════════════════════════════════════════
# Parse DEF file for SRAM placement coordinates
# ═══════════════════════════════════════════════════════════

def_file = '/home/wangduo/project/SpecLLMDSA/results/dsa_placed.def'

die_area = None
db_units_per_micron = 2000  # default
srams = []  # (name, x_um, y_um, w_um, h_um, type)
stdcell_count = 0
rows = []

with open(def_file, 'r') as f:
    in_components = False
    for line in f:
        line = line.strip()

        if line.startswith('UNITS DISTANCE MICRONS'):
            db_units_per_micron = int(line.split()[-2])

        if line.startswith('DIEAREA'):
            m = re.findall(r'\( (\d+) (\d+) \)', line)
            if len(m) == 2:
                x0 = int(m[0][0]) / db_units_per_micron
                y0 = int(m[0][1]) / db_units_per_micron
                x1 = int(m[1][0]) / db_units_per_micron
                y1 = int(m[1][1]) / db_units_per_micron
                die_area = (x0, y0, x1, y1)

        if line.startswith('ROW '):
            m = re.match(r'ROW \S+ \S+ (\d+) (\d+) \S+ DO (\d+)', line)
            if m:
                rx = int(m.group(1)) / db_units_per_micron
                ry = int(m.group(2)) / db_units_per_micron
                count = int(m.group(3))
                rows.append((rx, ry, count))

        if line.startswith('COMPONENTS'):
            in_components = True
            stdcell_count = int(line.split()[1])
            continue

        if line.startswith('END COMPONENTS'):
            in_components = False

        if in_components and '+ FIXED' in line or '+ FIRM' in line:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+(?:FIXED|FIRM)\s+\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m:
                name = m.group(1)
                cell_type = m.group(2)
                x = int(m.group(3)) / db_units_per_micron
                y = int(m.group(4)) / db_units_per_micron

                # Determine SRAM type
                if 'wt' in name:
                    stype = 'Weight'
                    color = '#2196F3'
                elif 'act' in name:
                    stype = 'Activation'
                    color = '#4CAF50'
                elif 'acc' in name:
                    stype = 'Accumulator'
                    color = '#FF9800'
                elif 'quantify' in name or 'buf' in name:
                    stype = 'Buffer'
                    color = '#9C27B0'
                else:
                    stype = 'Unknown'
                    color = '#757575'

                # Estimate size from cell type
                if 'scratchpad_1024x128' in cell_type:
                    w, h = 452.24, 380.715
                elif 'quantify' in cell_type.lower() or 'buffer' in cell_type.lower():
                    w, h = 500, 450  # estimate
                else:
                    w, h = 100, 100  # unknown

                srams.append({
                    'name': name, 'type': stype, 'color': color,
                    'x': x, 'y': y, 'w': w, 'h': h
                })

print(f"Die area: {die_area}")
print(f"SRAM macros: {len(srams)}")
print(f"StdCells (unplaced): {stdcell_count - len(srams)}")
print(f"Site rows: {len(rows)}")

# ═══════════════════════════════════════════════════════════
# Figure: 2-panel placement + area budget
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# ── Panel 1: SRAM macro placement with die outline ──
ax = axes[0]
ax.set_aspect('equal')

if die_area:
    x0, y0, x1, y1 = die_area
    die_w = x1 - x0
    die_h = y1 - y0
    ax.set_xlim(x0 - 100, x1 + 100)
    ax.set_ylim(y0 - 100, y1 + 100)

    # Die boundary
    rect = Rectangle((x0, y0), die_w, die_h, fill=False,
                      edgecolor='black', linewidth=2, linestyle='-')
    ax.add_patch(rect)

    # Core area (20um margin)
    core_rect = Rectangle((x0+20, y0+20), die_w-40, die_h-40, fill=False,
                          edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(core_rect)

    # Row grid hint
    if rows:
        for rx, ry, count in rows[:10]:  # sample first few rows
            ax.axhline(y=ry, color='lightgray', linewidth=0.3, alpha=0.5)

# Draw SRAMs
for s in srams:
    rect = FancyBboxPatch((s['x'], s['y']), s['w'], s['h'],
                           boxstyle="round,pad=4",
                           facecolor=s['color'], edgecolor='black',
                           linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    # Label smaller SRAMs with type
    if s['w'] < 500:
        ax.text(s['x'] + s['w']/2, s['y'] + s['h']/2, s['type'][:3],
                ha='center', va='center', fontsize=5, color='white', fontweight='bold')

# Annotate compute zone (between top and bottom SRAM rows)
sram_ys = [s['y'] for s in srams]
sram_y_max = max(s['y'] + s['h'] for s in srams)
sram_y_min = min(s['y'] for s in srams)

# Find the gap between bottom SRAMs and top SRAMs
srams_bottom = [s for s in srams if s['y'] < die_h/2]
srams_top = [s for s in srams if s['y'] > die_h/2]

if srams_bottom and srams_top:
    bottom_max_y = max(s['y'] + s['h'] for s in srams_bottom)
    top_min_y = min(s['y'] for s in srams_top)
    compute_zone = (x0+20, bottom_max_y + 20, die_w-40, top_min_y - bottom_max_y - 40)

    rect = Rectangle((compute_zone[0], compute_zone[1]),
                      compute_zone[2], compute_zone[3],
                      fill=True, facecolor='#E3F2FD', edgecolor='#1565C0',
                      linewidth=1.5, linestyle='--', alpha=0.4)
    ax.add_patch(rect)
    ax.text(x0 + die_w/2, bottom_max_y + compute_zone[3]/2,
            'COMPUTE ZONE\n' + f'{compute_zone[2]:.0f} × {compute_zone[3]:.0f} um',
            ha='center', va='center', fontsize=11,
            color='#1565C0', fontweight='bold', alpha=0.7)

legend_elements = [
    patches.Patch(facecolor='#2196F3', label=f'Weight SRAM (×8)'),
    patches.Patch(facecolor='#4CAF50', label=f'Activation SRAM (×8)'),
    patches.Patch(facecolor='#FF9800', label=f'Accumulator SRAM (×8)'),
    patches.Patch(facecolor='#9C27B0', label=f'Quantify Buffer (×1)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

ax.set_title(f'SpecLLM DSA — SRAM Macro Placement\nDie: {die_w:.0f} × {die_h:.0f} um ({die_w*die_h/1e6:.1f} mm²) | 25 SRAM Macros',
             fontsize=13, fontweight='bold')
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.grid(True, alpha=0.2)

# ── Panel 2: Area Breakdown ──
ax2 = axes[1]
ax2.axis('off')

# Area budget table
# Estimated areas from design spec
data = {
    'MsPE Array (64×64 DynLogic)':       (2.20, 0, '#E91E63'),
    'INT8 PE Array (32×32)':             (0.72, 0, '#FF5722'),
    'QuanRedPipe (64-ch FP32 Fusion)':   (0.50, 0, '#00BCD4'),
    'Quantify Engine':                    (0.15, 0, '#673AB7'),
    'Vector Unit (32-lane FP32)':        (0.025,0, '#009688'),
    'Controller + DMA + TL IF':          (0.10, 0, '#607D8B'),
    'Weight SRAM (8 × scratchpad 1K×128)': (0, 1.38, '#2196F3'),
    'Activation SRAM (8 × scratchpad)':    (0, 1.38, '#4CAF50'),
    'Accumulator SRAM (8 × scratchpad)':   (0, 1.38, '#FF9800'),
    'Quantify Buffer SRAM (128×1024)':     (0, 0.05, '#9C27B0'),
}

# Pie chart: StdCell vs SRAM
pie_ax = ax2.inset_axes([0.05, 0.55, 0.40, 0.40])
stdcell_total = sum(v[0] for v in data.values())
sram_total = sum(v[1] for v in data.values())
pie_ax.pie([stdcell_total, sram_total],
           labels=[f'StdCell\n{stdcell_total:.1f} mm²', f'SRAM\n{sram_total:.1f} mm²'],
           colors=['#FF7043', '#42A5F5'], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
pie_ax.set_title('Area Split', fontsize=10, fontweight='bold')

# Bar chart: module breakdown
bar_ax = ax2.inset_axes([0.50, 0.55, 0.48, 0.42])
names = list(data.keys())
stdcell_vals = [v[0] for v in data.values()]
sram_vals = [v[1] for v in data.values()]
colors = [v[2] for v in data.values()]

y_pos = range(len(names))
bar_ax.barh(y_pos, stdcell_vals, height=0.6, color=colors, alpha=0.8, label='StdCell')
bar_ax.barh(y_pos, sram_vals, height=0.6, left=stdcell_vals,
            color=colors, alpha=0.4, label='SRAM', hatch='//')
bar_ax.set_yticks(y_pos)
bar_ax.set_yticklabels([n.split('(')[0].strip() for n in names], fontsize=6)
bar_ax.set_xlabel('Area (mm²)', fontsize=8)
bar_ax.legend(fontsize=7, loc='lower right')
bar_ax.invert_yaxis()

# Summary text
summary = f"""Placement Summary
═══════════════
Die:        {die_w:.0f} × {die_h:.0f} um
Area:       {die_w*die_h/1e6:.2f} mm²

StdCells:   {stdcell_total:.1f} mm²
SRAM:       {sram_total:.1f} mm²
Active:     {stdcell_total+sram_total:.1f} mm²
Density:    {(stdcell_total+sram_total)/(die_w*die_h/1e6)*100:.1f}%

StdCell count: ~{stdcell_count - len(srams):,}
SRAM macros:   {len(srams)}
"""
ax2.text(0.05, 0.48, summary, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))

ax2.set_title('SpecLLM DSA — Area Breakdown', fontsize=13, fontweight='bold')

plt.tight_layout(pad=2)
outpath = '/home/wangduo/project/SpecLLMDSA/results/dsa_placement_heatmap.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'\nSaved: {outpath}')
