#!/usr/bin/env python3
"""SoC density heatmap: block names centered large, no area text, uniform density."""
import re, random, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch

random.seed(42)

def parse_def(path, x_off=0, y_off=0):
    cells = []; srams = []; db = 2000
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('UNITS'): db = int(line.split()[-2])
            if line.startswith('DIEAREA'):
                m = re.findall(r'\( (\d+) (\d+) \)', line)
                if len(m) == 2: die = tuple(int(v)/db for v in m[0] + m[1])
            if '+ FIXED' in line:
                m = re.match(r'\s*-\s*(\S+)\s+(\S+)\s+\+\s+FIXED\s+\(\s*(\d+)\s+(\d+)\s*\)', line)
                if m:
                    ctype = m.group(2); x = int(m.group(3))/db + x_off; y = int(m.group(4))/db + y_off
                    if 'scratchpad' in ctype: w, h = 452, 381
                    elif 'cc_bank' in ctype: w, h = 416, 392
                    elif 'cc_dir' in ctype: w, h = 452, 381
                    elif 'icache' in ctype or 'dcache' in ctype:
                        w, h = (115, 119) if 'data' in ctype else (78, 113)
                    else: continue
                    srams.append((x, y, w, h))
            if '+ PLACED' in line and 'LOGIC0' not in line:
                m = re.match(r'.*\(\s*(\d+)\s+(\d+)\s*\)', line)
                if m: cells.append((int(m.group(1))/db + x_off, int(m.group(2))/db + y_off))
    return cells, srams

# Layout
DSA_W, DSA_H = 3850, 3400
L2_W, L2_H = 4000, 1000; SB_W, SB_H = 500, 500; RT_W, RT_H = 1000, 1000

dsa_x, dsa_y = 0, 0
l2_x, l2_y = DSA_W + 100, 0
sb_x, sb_y = DSA_W + 100, L2_H + 50
rt_x = DSA_W + 100 + SB_W + 50
rt0_x, rt0_y = rt_x, L2_H + 50
rt1_x, rt1_y = rt_x + RT_W + 20, L2_H + 50
rt2_x, rt2_y = rt_x, L2_H + 50 + RT_H + 20
rt3_x, rt3_y = rt_x + RT_W + 20, L2_H + 50 + RT_H + 20

total_w = max(dsa_x+DSA_W, l2_x+L2_W, rt1_x+RT_W) + 60
total_h = max(dsa_y+DSA_H, L2_H+50+2*RT_H+20) + 60

# Parse all
all_cells = []; all_srams = []
for path, ox, oy in [
    ('/home/wangduo/project/SpecLLMDSA/results/dsa_placed_orfs.def', dsa_x, dsa_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/RocketTile_placed.def', rt0_x, rt0_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/RocketTile_placed.def', rt1_x, rt1_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/RocketTile_placed.def', rt2_x, rt2_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/RocketTile_placed.def', rt3_x, rt3_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/l2_placed_v14.def', l2_x, l2_y),
    ('/home/wangduo/project/sparsenn/results/cpu_st_top/sb_placed.def', sb_x, sb_y),
]:
    c, s = parse_def(path, ox, oy)
    all_cells += c; all_srams += s

# Render
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_aspect('equal')
ax.set_xlim(-20, total_w+20); ax.set_ylim(-20, total_h+20)

# Density heatmap - clip at lower percentile to reduce RT brightness
sample = all_cells if len(all_cells) <= 200000 else random.sample(all_cells, 200000)
xs = [p[0] for p in sample]; ys = [p[1] for p in sample]
h, xe, ye = np.histogram2d(xs, ys, bins=350, range=[[0, total_w], [0, total_h]])
h_sqrt = np.sqrt(h)
vmax = np.percentile(h_sqrt[h_sqrt > 0], 50)  # median clip for balance
ax.imshow(h_sqrt.T, origin='lower', extent=[0, total_w, 0, total_h],
          cmap='inferno', alpha=0.75, aspect='auto', interpolation='bilinear',
          vmax=vmax)

# SRAMs: dark gray
for x, y, w, h in all_srams:
    ax.add_patch(Rectangle((x, y), w, h, fill=True,
                  facecolor='#424242', edgecolor='#333', linewidth=0.3, alpha=1.0))

# Block outlines + large centered names
blocks = [
    (dsa_x, dsa_y, DSA_W, DSA_H, '#1565C0', 'NPU-Q\nDSA', 2.2),
    (l2_x, l2_y, L2_W, L2_H, '#4CAF50', 'L2\nCache', 1.5),
    (sb_x, sb_y, SB_W, SB_H, '#9C27B0', 'SB', 1.2),
    (rt0_x, rt0_y, RT_W, RT_H, '#FF9800', 'RT0', 1.0),
    (rt1_x, rt1_y, RT_W, RT_H, '#FFA726', 'RT1', 1.0),
    (rt2_x, rt2_y, RT_W, RT_H, '#FFB74D', 'RT2', 1.0),
    (rt3_x, rt3_y, RT_W, RT_H, '#FFCC80', 'RT3', 1.0),
]
for x, y, w, h, color, label, lw in blocks:
    ax.add_patch(FancyBboxPatch((x+2, y+2), w-4, h-4, boxstyle="round,pad=8",
                  facecolor='none', edgecolor=color, linewidth=lw, alpha=0.85))
    # Font size proportional but capped
    short_side = min(w, h)
    fontsize = min(short_side * 0.03, 60)  # 3% of short side, max 60pt
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=color,
            alpha=0.75)

# SoC boundary
ax.add_patch(Rectangle((0, 0), total_w, total_h, fill=False, edgecolor='black', linewidth=2.5))
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(pad=0.3)
plt.savefig('/home/wangduo/project/SpecLLMDSA/results/dsa_soc_placement.png', dpi=200, bbox_inches='tight')
print('SoC: {:.0f}x{:.0f} um, vmax={:.1f}'.format(total_w, total_h, vmax))
