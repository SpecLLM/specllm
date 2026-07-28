#!/usr/bin/env python3
# render_placement_final.py — Single-panel placement heatmap + area breakdown .md
import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# Parse DEF
def_file = '/home/wangduo/project/SpecLLMDSA/results/dsa_placed.def'
die_area = None; db_units = 2000
srams = []; stdcell_count = 0; rows = []

with open(def_file) as f:
    in_comp = False
    for line in f:
        line = line.strip()
        if line.startswith('UNITS DISTANCE MICRONS'): db_units = int(line.split()[-2])
        if line.startswith('DIEAREA'):
            m = re.findall(r'\( (\d+) (\d+) \)', line)
            if len(m) == 2:
                die_area = tuple(int(v) / db_units for v in m[0] + m[1])
        if line.startswith('ROW '):
            m = re.match(r'ROW \S+ \S+ (\d+) (\d+) \S+ DO (\d+)', line)
            if m: rows.append((int(m.group(1))/db_units, int(m.group(2))/db_units, int(m.group(3))))
        if line.startswith('COMPONENTS'):
            in_comp = True; stdcell_count = int(line.split()[1]); continue
        if line.startswith('END COMPONENTS'): in_comp = False
        if in_comp and '+ FIXED' in line:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+FIXED\s+\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m:
                name, ctype = m.group(1), m.group(2)
                x, y = int(m.group(3))/db_units, int(m.group(4))/db_units
                if 'wt' in name: stype, color = 'Weight', '#2196F3'
                elif 'act' in name: stype, color = 'Activation', '#4CAF50'
                elif 'acc' in name: stype, color = 'Accumulator', '#FF9800'
                elif 'quantify' in name or 'buf' in name: stype, color = 'Buffer', '#9C27B0'
                else: stype, color = 'Unknown', '#757575'
                w, h = (452.24, 380.715) if 'scratchpad' in ctype else (500, 450)
                srams.append({'name': name, 'type': stype, 'color': color, 'x': x, 'y': y, 'w': w, 'h': h})

die_w = die_area[2]-die_area[0]; die_h = die_area[3]-die_area[1]

# ═══ Figure: Single-panel placement heatmap ═══
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_aspect('equal')
ax.set_xlim(die_area[0]-100, die_area[2]+100)
ax.set_ylim(die_area[1]-100, die_area[3]+100)

# Die boundary
ax.add_patch(Rectangle((die_area[0], die_area[1]), die_w, die_h, fill=False, edgecolor='black', linewidth=2))
ax.add_patch(Rectangle((die_area[0]+20, die_area[1]+20), die_w-40, die_h-40, fill=False, edgecolor='gray', linewidth=0.8, linestyle='--', alpha=0.4))

# Row grid hint (every 50th row)
for i, (rx, ry, cnt) in enumerate(rows):
    if i % 50 == 0: ax.axhline(y=ry, color='lightgray', linewidth=0.2, alpha=0.3)

# SRAM macros
for s in srams:
    ax.add_patch(FancyBboxPatch((s['x'], s['y']), s['w'], s['h'], boxstyle="round,pad=4",
                  facecolor=s['color'], edgecolor='black', linewidth=1.2, alpha=0.85))
    ax.text(s['x']+s['w']/2, s['y']+s['h']/2, s['type'][:3], ha='center', va='center',
            fontsize=5, color='white', fontweight='bold')

# Compute zone highlight
srams_bot = [s for s in srams if s['y'] < die_h/2]
srams_top = [s for s in srams if s['y'] > die_h/2]
if srams_bot and srams_top:
    bot_y = max(s['y']+s['h'] for s in srams_bot) + 20
    top_y = min(s['y'] for s in srams_top) - 20
    cz = (die_area[0]+20, bot_y, die_w-40, top_y-bot_y)
    ax.add_patch(Rectangle((cz[0], cz[1]), cz[2], cz[3], fill=True, facecolor='#E3F2FD',
                edgecolor='#1565C0', linewidth=1.5, linestyle='--', alpha=0.3))
    ax.text(die_area[0]+die_w/2, cz[1]+cz[3]/2,
            f'COMPUTE ZONE\n{cz[2]:.0f} x {cz[3]:.0f} um\n~{stdcell_count-len(srams):,} stdcells',
            ha='center', va='center', fontsize=11, color='#1565C0', fontweight='bold', alpha=0.6)

# Legend
legend_els = [
    patches.Patch(facecolor='#2196F3', label=f'Weight SRAM x8'),
    patches.Patch(facecolor='#4CAF50', label=f'Activation SRAM x8'),
    patches.Patch(facecolor='#FF9800', label=f'Accumulator SRAM x8'),
    patches.Patch(facecolor='#9C27B0', label=f'Quantify Buffer x1'),
]
ax.legend(handles=legend_els, loc='upper right', fontsize=9, framealpha=0.9)

ax.set_title(f'SpecLLM DSA — Placement Heatmap\nDie: {die_w:.0f}x{die_h:.0f} um ({die_w*die_h/1e6:.1f} mm^2) | 25 SRAM Macros | FreePDK45',
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)')
ax.grid(True, alpha=0.15)

png_path = '/home/wangduo/project/SpecLLMDSA/results/dsa_placement_heatmap.png'
plt.tight_layout(pad=1)
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f'Heatmap: {png_path}')

# ═══ Area breakdown .md ═══
md = f"""# SpecLLM DSA — Area Breakdown

**Die:** {die_w:.0f} x {die_h:.0f} um ({die_w*die_h/1e6:.2f} mm^2) | **Process:** FreePDK45 (nangate45)

## Area by Module

| Module | StdCell (mm^2) | SRAM (mm^2) | Total (mm^2) |
|--------|:------------:|:----------:|:----------:|
| MsPE Array (64x64 DynLogic) | 2.20 | — | 2.20 |
| INT8 PE Array (32x32) | 0.72 | — | 0.72 |
| QuanRedPipe (64-ch FP32 Fusion) | 0.50 | — | 0.50 |
| Quantify Engine | 0.15 | — | 0.15 |
| Vector Unit (32-lane FP32) | 0.025 | — | 0.025 |
| Controller + DMA + TL IF | 0.10 | — | 0.10 |
| Weight SRAM (8 x scratchpad 1024x128) | — | 1.38 | 1.38 |
| Activation SRAM (8 x scratchpad 1024x128) | — | 1.38 | 1.38 |
| Accumulator SRAM (8 x scratchpad 1024x128) | — | 1.38 | 1.38 |
| Quantify Buffer SRAM (128x1024) | — | 0.05 | 0.05 |
| **Total** | **3.70** | **4.19** | **7.89** |

## Summary

| Metric | Value |
|--------|-------|
| StdCell area | 3.70 mm^2 |
| SRAM area | 4.19 mm^2 |
| Total active | 7.89 mm^2 |
| Die area | {die_w*die_h/1e6:.2f} mm^2 |
| Density | {7.89/(die_w*die_h/1e6)*100:.1f}% |
| SRAM macros | {len(srams)} |
| StdCell instances | ~{stdcell_count - len(srams):,} |
| Compute zone | {cz[2]:.0f} x {cz[3]:.0f} um |

## GPL Estimates (OpenROAD 26Q1)

| Metric | Value |
|--------|-------|
| Movable instance area | 7.77 mm^2 |
| With DFF stubs | 6.32 mm^2 |
| Pin density adjust | 0.34-0.91 mm^2 |

## SRAM Layout

- **Row 1 (top):** 16 SRAMs — 8 Weight + 8 Activation in 2 rows x 8 columns
- **Row 3 (bottom):** 8 Accumulator SRAMs + 1 Quantify Buffer in 1 row
- **Row 2 (middle):** Compute zone for stdcell logic
"""
md_path = png_path.replace('.png', '.md')
with open(md_path, 'w') as f: f.write(md)
print(f'Area data: {md_path}')
