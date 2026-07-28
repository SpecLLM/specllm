#!/usr/bin/env python3
# render_dsa_soc_placement.py — DSA + SoC Placement Visualization
# Renders the 4-RocketTile + NPU-Q DSA SoC floorplan with area heatmap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# ═══════════════════════════════════════════════════════════
# Area Data (from sparsenn reference + DSA estimates)
# ═══════════════════════════════════════════════════════════

# SoC-level modules
SOC_MODULES = {
    'NPU-Q DSA':      {'w': 5000, 'h': 3500, 'color': '#2196F3', 'area_mm2': 17.5},
    'RocketTile 0':   {'w': 1000, 'h': 1000, 'color': '#FF9800', 'area_mm2': 1.0},
    'RocketTile 1':   {'w': 1000, 'h': 1000, 'color': '#FF9800', 'area_mm2': 1.0},
    'RocketTile 2':   {'w': 1000, 'h': 1000, 'color': '#FF9800', 'area_mm2': 1.0},
    'RocketTile 3':   {'w': 1000, 'h': 1000, 'color': '#FF9800', 'area_mm2': 1.0},
    'L2 Cache':       {'w': 4000, 'h': 1000, 'color': '#4CAF50', 'area_mm2': 4.0},
    'SystemBus':      {'w': 500,  'h': 500,  'color': '#9C27B0', 'area_mm2': 0.25},
}

# DSA internal sub-modules
DSA_MODULES = {
    'MsPE Array\n(64×64 DynLogic)':    {'w': 1700, 'h': 1500, 'color': '#E91E63', 'stdcell_mm2': 2.2},
    'INT8 PE Array\n(32×32)':          {'w':  700, 'h': 1000, 'color': '#FF5722', 'stdcell_mm2': 0.72},
    'QuanRedPipe\n(64-ch FP32 Fusion)': {'w': 1700, 'h':  300, 'color': '#00BCD4', 'stdcell_mm2': 0.5},
    'Quantify\nEngine':                 {'w':  400, 'h':  300, 'color': '#673AB7', 'stdcell_mm2': 0.15},
    'Vector Unit\n(32-lane FP32)':      {'w':  300, 'h':  300, 'color': '#009688', 'stdcell_mm2': 0.025},
    'Ctrl + DMA\n+ TL IF':             {'w':  300, 'h':  300, 'color': '#607D8B', 'stdcell_mm2': 0.1},
    'SRAM Bank\n(24 macros)':          {'w': 3000, 'h': 1800, 'color': '#FFC107', 'stdcell_mm2': 0, 'sram_mm2': 4.2},
}

# ═══════════════════════════════════════════════════════════
# Figure 1: SoC Top-Level Floorplan
# ═══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# --- SoC Top-Level ---
ax = axes[0]
ax.set_xlim(0, 6000)
ax.set_ylim(0, 6000)
ax.set_aspect('equal')
ax.set_title('SpecLLM SoC: 4×RocketTile + NPU-Q DSA\nChip: ~5400×5500 um ≈ 29.7 mm² (FreePDK45)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.grid(True, alpha=0.3, linestyle='--')

# Place DSA (top, full width)
dsa = SOC_MODULES['NPU-Q DSA']
dsa_x, dsa_y = 200, 2000
rect = FancyBboxPatch((dsa_x, dsa_y), dsa['w'], dsa['h'],
                       boxstyle="round,pad=8", facecolor=dsa['color'],
                       edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(rect)
ax.text(dsa_x + dsa['w']/2, dsa_y + dsa['h']/2,
        f"NPU-Q DSA\n{dsa['w']}×{dsa['h']} um\n{dsa['area_mm2']} mm²",
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Place L2 (bottom-left)
l2 = SOC_MODULES['L2 Cache']
l2_x, l2_y = 200, 500
rect = FancyBboxPatch((l2_x, l2_y), l2['w'], l2['h'],
                       boxstyle="round,pad=8", facecolor=l2['color'],
                       edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(rect)
ax.text(l2_x + l2['w']/2, l2_y + l2['h']/2,
        f"L2 Cache\n{l2['w']}×{l2['h']} um\n{l2['area_mm2']} mm²",
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Place SystemBus
sb = SOC_MODULES['SystemBus']
sb_x, sb_y = 4200, 750
rect = FancyBboxPatch((sb_x, sb_y), sb['w'], sb['h'],
                       boxstyle="round,pad=6", facecolor=sb['color'],
                       edgecolor='black', linewidth=2, alpha=0.85)
ax.add_patch(rect)
ax.text(sb_x + sb['w']/2, sb_y + sb['h']/2, 'SB\n0.25',
        ha='center', va='center', fontsize=7, fontweight='bold', color='white')

# Place 4 RocketTiles
rt_colors = ['#FF9800', '#FFA726', '#FFB74D', '#FFCC80']
for i in range(4):
    rt = SOC_MODULES[f'RocketTile {i}']
    if i < 2:
        rx = 4200 + 550 + (i * 1100)
        ry = 750
    else:
        rx = 4200 + 550 + ((i-2) * 1100)
        ry = 750 - 1100

    rect = FancyBboxPatch((rx, ry), rt['w'], rt['h'],
                           boxstyle="round,pad=6", facecolor=rt_colors[i],
                           edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(rx + rt['w']/2, ry + rt['h']/2, f'RT{i}\n1.0 mm²',
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')

# Interconnect lines
bus_center_x = sb_x + sb['w']/2
bus_center_y = sb_y + sb['h']/2
# Bus to DSA
ax.plot([bus_center_x, dsa_x + dsa['w']/2], [bus_center_y, dsa_y + dsa['h']],
        'k--', linewidth=0.8, alpha=0.5)
# Bus to L2
ax.plot([bus_center_x, l2_x + l2['w']/2], [bus_center_y, l2_y + l2['h']/2],
        'k--', linewidth=0.8, alpha=0.5)
# Bus to RocketTiles
for i in range(4):
    rx = 4200 + 550 + ((i%2) * 1100) + 500
    ry = 750 + (0 if i < 2 else -1100) + 500
    ax.plot([bus_center_x, rx], [bus_center_y, ry],
            'k--', linewidth=0.8, alpha=0.5)

# Area legend
area_text = f"""SoC Area Breakdown:
├─ NPU-Q DSA:     17.50 mm² (58.8%)
├─ L2 Cache:       4.00 mm² (13.4%)
├─ 4× RocketTile:  4.00 mm² (13.4%)
├─ SystemBus:      0.25 mm²  (0.8%)
└─ Routing/Waste:  ~4.00 mm² (13.5%)
Total Die:        ~29.70 mm²"""
ax.text(0.02, 0.98, area_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- DSA Internal Placement ---
ax2 = axes[1]
ax2.set_xlim(0, 5800)
ax2.set_ylim(0, 4200)
ax2.set_aspect('equal')
ax2.set_title('NPU-Q DSA Internal Floorplan\nDie: ~5000×3500 um ≈ 17.5 mm² | Active: ~7.9 mm² (45% density)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('X (um)')
ax2.set_ylabel('Y (um)')
ax2.grid(True, alpha=0.3, linestyle='--')

# Place sub-modules with estimated positions
placements = [
    # (name, x, y, w, h, color, label)
    ('MsPE Array\n(64×64 DynLogic)',   200, 1200, 1700, 1500, '#E91E63', '2.2 mm² stdcell\nQ2K 4-lane | Q3K 2-lane'),
    ('INT8 PE Array\n(32×32)',         1950, 1600,  700, 1000, '#FF5722', '0.72 mm² stdcell\n1024 PE × 701 um²'),
    ('QuanRedPipe\n(FP32 Fusion)',      200,  800, 1700,  300, '#00BCD4', '0.5 mm²\n64-ch MAC-Adder\n2-cycle pipe'),
    ('Quantify\nEngine',                1950, 1150,  400,  350, '#673AB7', '0.15 mm²\nO(log N) tree'),
    ('Vector Unit\n(32-lane)',          2400, 1150,  300,  350, '#009688', '0.025 mm²\nReLU/Softmax/LN'),
    ('Ctrl+DMA\n+TL IF',                1950, 750,   700,  350, '#607D8B', '0.1 mm²\nFSM+CSR+DMA'),
    ('SRAM Bank\n(24× scratchpad)',     200,  200, 3000, 1800, '#FFC107', '4.2 mm² SRAM\n8wt+8act+8acc'),
]

for name, x, y, w, h, color, label in placements:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=6",
                           facecolor=color, edgecolor='black',
                           linewidth=1.5, alpha=0.85)
    ax2.add_patch(rect)
    ax2.text(x + w/2, y + h/2 + h*0.08, name, ha='center', va='center',
             fontsize=8, fontweight='bold', color='white')
    ax2.text(x + w/2, y + h/2 - h*0.20, label, ha='center', va='center',
             fontsize=6, color='white', alpha=0.9)

# Draw data flow arrows
arrows = [
    (1050, 1200, 1050, 1100, '↓'),   # MsPE → QuanRedPipe
    (2300, 1600, 1050, 1100, '↙'),   # INT8 → QuanRedPipe
    (1050, 800, 1050, 500, '↓'),     # QuanRedPipe → VecUnit/SRAM
    (2150, 1150, 2150, 950, '↓'),    # QuantE → SRAM
]
for x1, y1, x2, y2, _ in arrows:
    ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

# SRAM macro grid inside SRAM bank
for row in range(4):
    for col in range(6):
        sx = 250 + col * 470
        sy = 250 + row * 420
        rect = Rectangle((sx, sy), 440, 380, fill=False,
                         edgecolor='#FFC107', linewidth=0.5, alpha=0.6)
        ax2.add_patch(rect)

# Area pie chart (inset)
pie_ax = ax2.inset_axes([0.55, 0.55, 0.40, 0.40])
dims_labels = ['MsPE (2.2)', 'INT8 PE (0.72)', 'QuanRed (0.5)',
               'QuantE (0.15)', 'VecUnit (0.025)', 'Ctrl (0.1)', 'SRAM (4.2)']
dims_sizes = [2.2, 0.72, 0.5, 0.15, 0.025, 0.1, 4.2]
dims_colors = ['#E91E63', '#FF5722', '#00BCD4', '#673AB7', '#009688', '#607D8B', '#FFC107']
patches_pie, texts = pie_ax.pie(dims_sizes, colors=dims_colors,
                                  startangle=90,
                                  textprops={'fontsize': 5})
pie_ax.set_title('Area by Module (mm²)', fontsize=7)

# DSA summary text
dsa_text = f"""DSA Active Area: ~7.9 mm²
├─ StdCell Logic:  ~3.7 mm² (46.8%)
│   ├─ MsPE Array:      2.20 mm²
│   ├─ INT8 PE Array:   0.72 mm²
│   ├─ QuanRedPipe:     0.50 mm²
│   ├─ Quantify Engine:  0.15 mm²
│   ├─ Vector Unit:     0.025 mm²
│   └─ Ctrl+DMA+TL IF:  0.10 mm²
└─ SRAM:           ~4.2 mm² (53.2%)
    ├─ Weight (8×):      1.4 mm²
    ├─ Activation (8×):  1.4 mm²
    └─ Accumulator (8×): 1.4 mm²"""
ax2.text(0.02, 0.98, dsa_text, transform=ax2.transAxes, fontsize=6.5,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout(pad=2)
plt.savefig('/home/wangduo/project/SpecLLMDSA/results/placement_dsa_soc.png',
            dpi=150, bbox_inches='tight')
print("Placement diagram saved to results/placement_dsa_soc.png")
