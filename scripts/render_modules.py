#!/usr/bin/env python3
"""DSA density heatmap: dark gray SRAMs, pure sqrt density."""
import re, random, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np
from matplotlib.patches import Rectangle

random.seed(42)
def_path = '/home/wangduo/project/SpecLLMDSA/results/dsa_placed_orfs.def'
db = 2000; srams = []; cells = []

with open(def_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('UNITS'): db = int(line.split()[-2])
        if line.startswith('DIEAREA'):
            m = re.findall(r'\( (\d+) (\d+) \)', line)
            if len(m) == 2: die = tuple(int(v)/db for v in m[0] + m[1])
        if '+ FIXED' in line:
            m = re.match(r'.*\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m: srams.append((int(m.group(1))/db, int(m.group(2))/db))
        if '+ PLACED' in line and 'scratchpad' not in line and 'LOGIC0' not in line:
            m = re.match(r'.*\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m: cells.append((int(m.group(1))/db, int(m.group(2))/db))

dw = die[2]-die[0]; dh = die[3]-die[1]

sample = cells if len(cells) <= 150000 else random.sample(cells, 150000)
xs = [c[0] for c in sample]; ys = [c[1] for c in sample]

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_aspect('equal')
ax.set_xlim(-20, dw+20); ax.set_ylim(-20, dh+20)

# Density heatmap
h, xe, ye = np.histogram2d(xs, ys, bins=300, range=[[0, dw], [0, dh]])
h_sqrt = np.sqrt(h)
vmax = np.percentile(h_sqrt[h_sqrt > 0], 50)  # median clip for balanced look
ax.imshow(h_sqrt.T, origin='lower', extent=[0, dw, 0, dh],
          cmap='inferno', alpha=0.80, aspect='auto', interpolation='bilinear',
          vmax=vmax)

# SRAMs: dark gray, opaque
for sx, sy in srams:
    ax.add_patch(Rectangle((sx, sy), 452, 381, fill=True,
                  facecolor='#424242', edgecolor='#212121', linewidth=1.0, alpha=1.0))

ax.add_patch(Rectangle((0, 0), dw, dh, fill=False, edgecolor='black', linewidth=2.5))
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(pad=0.3)
plt.savefig('/home/wangduo/project/SpecLLMDSA/results/dsa_placement_heatmap.png', dpi=200, bbox_inches='tight')
print('DSA: {:.0f}x{:.0f} um, {} cells, {} SRAMs'.format(dw, dh, len(cells), len(srams)))
