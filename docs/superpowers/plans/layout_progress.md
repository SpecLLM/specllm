# specLLM_dsa_top — Synthesis & Layout Progress

**Start:** 2026-07-08 | **Updated:** 2026-07-10
**Status:** ORFS synth done, RTL fixes for array optimization in progress

---

## Current State

| Step | Status | Time | Result |
|------|:--:|------|--------|
| RTL fixes (vector_reg + FP16 array + INT8 removal) | ✅ | — | 3 new + 4 modified files |
| ORFS canonicalize | ✅ | 3:07 | 0 errors, 51MB RTLIL |
| ORFS synth | ✅ | 9:04 | 0 errors, 68MB netlist |
| Array verification | ⚠️ | — | mspe 3072/4096, fp16_mul 1024/1024 |
| ORFS do-1_synth + GPL | ⏳ | — | pending netlist cleanup |

---

## RTL Changes (2026-07-10)

### New Files
| File | Lines | Description |
|------|:----:|-------------|
| `rtl/vector_reg.v` | 123 | 64-ch buffer register, double-batch VecUnit |
| `rtl/fp16_pe.v` | 79 | FP16 MAC PE (fp16_mul + fp32_add) |
| `rtl/fp16_array_32x32.v` | 99 | 32x32 FP16 systolic array |

### Modified Files
| File | Change |
|------|--------|
| `specLLM_dsa_top.v` | vector_reg + fp16_array instantiation, acc_wr_data mux for fp16 output |
| `dsa_controller.v` | Added S_FP16_COMPUTE state, fp16_array_start/done ports |
| `dsa_params.vh` | Added FP16_PE_ROWS/COLS, DSA_STATES=11 |
| `fp16_mul.v` | Ported from sparsenn |

### Added to SYNTH_BLACKBOXES
`fp16_mul` (pre-synthesized bb_fp16_mul.v, 93KB)

---

## Array Instance Counts (Post-Synth)

| Array | RTL Spec | Netlist Actual | % Preserved |
|-------|:--------:|:------------:|:----------:|
| MsPE (mspe_cell) | 64×64=4096 | 64×48=3072 | 75% (↑ from 25%) |
| INT8 (pe_int8) | 32×32=1024 | 32×16=512 | 50% |
| FP16 (fp16_mul) | 32×32=1024 | 1024 | **100%** ✅ |

**MsPE improvement:** vector_reg doubled columns (16→48) by making all 64 QuanRedPipe outputs observable through the buffer. Remaining 16 columns (col[48-63]) need VecUnit second-batch trigger.

---

## ORFS Flow (Working Recipe)

```bash
# 1. Canonicalize (only needed after RTL change)
docker exec -w /flow/flow gemmini-flow make \
  DESIGN_CONFIG=./designs/nangate45/specLLM_dsa_top/config.mk do-yosys-canonicalize

# 2. Synth (custom synth_fixed.tcl skips keep_hierarchy -min_cost)
docker exec -w /flow/flow gemmini-flow make \
  DESIGN_CONFIG=./designs/nangate45/specLLM_dsa_top/config.mk do-yosys

# 3. Clean netlist for STA
docker exec -w /flow/flow gemmini-flow python3 /flow/flow/dsa_work/clean_netlist2.py

# 4. ODB + floorplan + GPL
docker exec -w /flow/flow gemmini-flow make \
  DESIGN_CONFIG=./designs/nangate45/specLLM_dsa_top/config.mk do-1_synth

# 5. Manual GPL with SRAM placement
docker exec -i gemmini-flow /usr/bin/openroad -no_init -exit \
  /flow/flow/dsa_work/compact_gpl.tcl
```

---

## Key Files

| Path | Purpose |
|------|---------|
| `rtl/` | 23 RTL files (20 original + 3 new) |
| `scripts/synth/synth_dsa.ys` | Standalone Yosys synthesis |
| `gemmini-flow/flow/designs/nangate45/specLLM_dsa_top/config.mk` | ORFS config |
| `gemmini-flow/flow/designs/nangate45/specLLM_dsa_top/synth_fixed.tcl` | Custom synth (no keep_hierarchy -min_cost) |
| `gemmini-flow/flow/dsa_work/clean_netlist2.py` | STA netlist cleaner |
| `gemmini-flow/flow/dsa_work/compact_gpl.tcl` | GPL with -skip_io + SRAM placement |
| `results/dsa_placement_heatmap.png` | Latest placement heatmap |
| `results/dsa_placed_orfs.def` | Latest placed DEF |
