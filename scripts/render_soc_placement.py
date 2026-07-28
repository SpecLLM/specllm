#!/usr/bin/env python3
"""Render SoC-level placement: 4×RocketTile + DSA + L2 + SystemBus."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np

# ═══ Block Definitions ═══
# DSA: from our GPL (6400×3200)
# RocketTile: 1000×1000 (sparsenn LEF)
# L2: 4000×1000 (sparsenn)
# SystemBus: 500×500 (sparsenn)

blocks = {
    'NPU-Q DSA':    {'w': 6400, 'h': 3200, 'color': '#1565C0', 'label': 'NPU-Q DSA\n6400×3200\n20.5 mm²\n189K cells'},
    'RocketTile 0': {'w': 1000, 'h': 1000, 'color': '#FF9800', 'label': 'RT0\n1.0 mm²'},
    'RocketTile 1': {'w': 1000, 'h': 1000, 'color': '#FFA726', 'label': 'RT1'},
    'RocketTile 2': {'w': 1000, 'h': 1000, 'color': '#FFB74D', 'label': 'RT2'},
    'RocketTile 3': {'w': 1000, 'h': 1000, 'color': '#FFCC80', 'label': 'RT3'},
    'L2 Cache':     {'w': 4000, 'h': 1000, 'color': '#4CAF50', 'label': 'L2 Cache\n4000×1000\n4.0 mm²'},
    'SystemBus':    {'w': 500,  'h': 500,  'color': '#9C27B0', 'label': 'SB\n0.25'},
}

# ═══ Layout ═══
# Top: DSA (6400 wide)
# Bottom: L2 (4000) + SB (500) + RT0-3 in 2 rows of 2 (2000×1000 each row)
# RT2+RT3 stacked below RT0+RT1

dsa_x, dsa_y = 0, 2200
l2_x, l2_y = 0, 500
sb_x, sb_y = 4000, 750
rt0_x, rt0_y = 4500, 750
rt1_x, rt1_y = 5500, 750
rt2_x, rt2_y = 4500, 500 - 1050
rt3_x, rt3_y = 5500, 500 - 1050

total_w = 6500  # 4000 L2 + 500 SB + 2000 RT
total_h = 3200 + 2200  # DSA (3200) + bottom row (2200)

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_aspect('equal')
ax.set_xlim(-200, total_w + 200)
ax.set_ylim(-200, total_h + 200)

# ═══ Draw blocks ═══
positions = {
    'NPU-Q DSA':    (dsa_x, dsa_y),
    'L2 Cache':     (l2_x, l2_y),
    'SystemBus':    (sb_x, sb_y),
    'RocketTile 0': (rt0_x, rt0_y),
    'RocketTile 1': (rt1_x, rt1_y),
    'RocketTile 2': (rt2_x, rt2_y),
    'RocketTile 3': (rt3_x, rt3_y),
}

for name, (x, y) in positions.items():
    b = blocks[name]
    rect = FancyBboxPatch((x, y), b['w'], b['h'], boxstyle="round,pad=10",
                           facecolor=b['color'], edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + b['w']/2, y + b['h']/2, b['label'], ha='center', va='center',
            fontsize=10 if 'DSA' in name else 9, fontweight='bold', color='white')

# ═══ DSA internal detail (from our GPL) ═══
# SRAM rows overlaid on DSA
SRAM_W, SRAM_H = 452, 381
GAP = 20
# Bottom row (weight SRAMs)
for i in range(8):
    sx = dsa_x + 20 + i * (SRAM_W + GAP)
    sy = dsa_y + 40
    ax.add_patch(Rectangle((sx, sy), SRAM_W, SRAM_H, fill=True,
                  facecolor='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.7))
# Top rows
for row in range(2):
    for i in range(8):
        sx = dsa_x + 20 + i * (SRAM_W + GAP)
        sy = dsa_y + 3200 - 40 - SRAM_H - row * (SRAM_H + GAP)
        color = '#4CAF50' if row == 0 else '#FF9800'
        ax.add_patch(Rectangle((sx, sy), SRAM_W, SRAM_H, fill=True,
                      facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7))

# Compute zone outline
cz_y0 = dsa_y + 40 + SRAM_H + GAP
cz_y1 = dsa_y + 3200 - 40 - 2*SRAM_H - 2*GAP
ax.add_patch(Rectangle((dsa_x + 20, cz_y0), 6400-40, cz_y1-cz_y0,
              fill=False, edgecolor='white', linewidth=2, linestyle='--', alpha=0.6))
ax.text(dsa_x + 3200, cz_y0 + (cz_y1-cz_y0)/2,
        'COMPUTE ZONE\n~189K cells', ha='center', va='center',
        fontsize=9, color='white', fontweight='bold', alpha=0.7)

# ═══ Interconnect lines ═══
bus_cx = sb_x + 250
bus_cy = sb_y + 250
# Bus → DSA
ax.plot([bus_cx, dsa_x + 3200], [bus_cy, dsa_y + 3200], 'k--', linewidth=0.8, alpha=0.4)
# Bus → L2
ax.plot([bus_cx, l2_x + 2000], [bus_cy, l2_y + 500], 'k--', linewidth=0.8, alpha=0.4)
# Bus → RocketTiles
for rx, ry in [(rt0_x+500, rt0_y+500), (rt1_x+500, rt1_y+500),
                (rt2_x+500, rt2_y+500), (rt3_x+500, rt3_y+500)]:
    ax.plot([bus_cx, rx], [bus_cy, ry], 'k--', linewidth=0.8, alpha=0.4)

# ═══ Title and labels ═══
ax.set_title('SpecLLM SoC: 4×RocketTile + NPU-Q DSA + L2 Cache\n'
             + 'Chip: {:.0f}×{:.0f} um ({:.1f} mm²) | FreePDK45 (nangate45)'.format(
                 total_w, total_h, total_w*total_h/1e6),
             fontsize=14, fontweight='bold')

# Area breakdown
area_text = (
    'SoC Area Breakdown:\n'
    '  NPU-Q DSA:      20.5 mm² (62%)\n'
    '  4×RocketTile:    4.0 mm² (12%)\n'
    '  L2 Cache:        4.0 mm² (12%)\n'
    '  SystemBus:       0.25 mm² (1%)\n'
    '  Routing/Waste:  ~4.5 mm² (13%)\n'
    '  Total Die:      ~33.2 mm²'
)
ax.text(0.02, 0.98, area_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Legend
leg = [
    mpatches.Patch(facecolor='#1565C0', label='NPU-Q DSA'),
    mpatches.Patch(facecolor='#FF9800', label='RocketTile (×4)'),
    mpatches.Patch(facecolor='#4CAF50', label='L2 Cache'),
    mpatches.Patch(facecolor='#9C27B0', label='SystemBus'),
]
ax.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout(pad=1)
out = '/home/wangduo/project/SpecLLMDSA/results/dsa_soc_placement.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved: {} ({:.0f} KB)'.format(out, plt.gcf().get_size_inches().prod() * 150 * 10))
PYEOF
