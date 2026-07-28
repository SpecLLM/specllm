#!/usr/bin/env python3
import re, random, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch; import matplotlib.patches as mpatches

random.seed(42)
def_path = '/home/wangduo/project/SpecLLMDSA/results/dsa_placed_orfs.def'
die = None; db_units = 2000; stdcell_pts = []; srams = []; n_total = 0

with open(def_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('UNITS'): db_units = int(line.split()[-2])
        if line.startswith('DIEAREA'):
            m = re.findall(r'\( (\d+) (\d+) \)', line)
            if len(m) == 2: die = tuple(int(v)/db_units for v in m[0] + m[1])
        if '+ FIXED' in line:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+FIXED\s+\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m:
                name, ctype = m.group(1), m.group(2)
                x, y = int(m.group(3))/db_units, int(m.group(4))/db_units
                w = 452.24 if 'scratchpad' in ctype else 500
                h = 380.715 if 'scratchpad' in ctype else 450
                color = '#2196F3' if 'wt' in name.lower() else ('#4CAF50' if 'act' in name.lower() else '#FF9800')
                label = 'W' if 'wt' in name.lower() else ('A' if 'act' in name.lower() else 'C')
                srams.append((name, x, y, w, h, color, label))
        if '+ PLACED' in line and 'scratchpad' not in line and 'LOGIC0' not in line:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m:
                x, y = int(m.group(3))/db_units, int(m.group(4))/db_units
                stdcell_pts.append((x, y)); n_total += 1

dw = die[2]-die[0]; dh = die[3]-die[1]
print('Die: {:.0f}x{:.0f} um ({:.1f} mm2), SRAM: {}, StdCell: {}'.format(dw, dh, dw*dh/1e6, len(srams), n_total))

sample = stdcell_pts if n_total <= 50000 else random.sample(stdcell_pts, 50000)
xs = [p[0] for p in sample]; ys = [p[1] for p in sample]

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_aspect('equal'); ax.set_xlim(-80, dw+80); ax.set_ylim(-80, dh+80)
h, xe, ye = np.histogram2d(xs, ys, bins=120, range=[[0, dw], [0, dh]])
ax.imshow(np.sqrt(h.T), origin='lower', extent=[0, dw, 0, dh],
          cmap='inferno', alpha=0.8, aspect='auto', interpolation='bilinear')
ax.add_patch(Rectangle((0, 0), dw, dh, fill=False, edgecolor='black', linewidth=2))

for name, x, y, w, h, color, label in srams:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=3",
                  facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.9))
    if w > 200:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=6, color='white', fontweight='bold')

overlaps = sum(1 for sx, sy in stdcell_pts
               for n, rx, ry, rw, rh, _, _ in srams
               if rx<=sx<=rx+rw and ry<=sy<=ry+rh)
print('Overlaps: {} / {} ({:.2f}%)'.format(overlaps, n_total, 100*overlaps/n_total))

ax.set_title('SpecLLM DSA — GPL Placement (Compact Die)\nDie: {:.0f}x{:.0f} um ({:.1f} mm2) | {:,} stdcells | 24 SRAM | FreePDK45'.format(
    dw, dh, dw*dh/1e6, n_total), fontsize=13, fontweight='bold')
ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)')
ax.legend(handles=[
    mpatches.Patch(facecolor='#2196F3', label='Weight SRAM (x8)'),
    mpatches.Patch(facecolor='#4CAF50', label='Activation SRAM (x8)'),
    mpatches.Patch(facecolor='#FF9800', label='Accumulator SRAM (x8)')],
    loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout(pad=1)
plt.savefig('/home/wangduo/project/SpecLLMDSA/results/dsa_placement_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved')
