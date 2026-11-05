# config_dsa.mk — specLLM_dsa_top ORFS synthesis config
# Reference: /home/user/project/sparsenn/cpu_st_top/st_v20/config.mk
#            /home/user/project/sparsenn/docs/superpowers/plans/synth.md

export DESIGN_NAME = specLLM_dsa_top
export PLATFORM    = nangate45

# ── Synthesis frontend ──
# Use standalone Yosys script (synth_dsa.ys) with read_verilog -sv
# for SystemVerilog +: construct support and #(DELAY) tolerance
export SYNTH_HIERARCHICAL = 0
export SYNTH_SCRIPT = $(DESIGN_HOME)/$(PLATFORM)/scripts/synth/synth_dsa.ys
export SYNTH_HDL_FRONTEND = verilog

# ── Blackbox modules (FP32 + SRAM + PE cells) ──
# Pre-synthesized in synth_dsa.ys Phase 0-1, loaded via read_verilog -lib
export SYNTH_BLACKBOXES = scratchpad_1024x128_freepdk45 \
                           quantify_buffer_sram \
                           fp32_mul fp32_add fp32_exp fp32_gelu fp32_sqrt \
                           mspe_cell

# ── SRAM LEF (from sparsenn) ──
export ADDITIONAL_LEFS = $(DESIGN_HOME)/$(PLATFORM)/cpu_st_top/st_block_hier/scratchpad_1024x128_freepdk45.lef

# ── Floorplan ──
# Signoff: design area 13.84 mm² (13.75 macro + 0.086 logic)
# Die: 4200×4300 um = 18.06 mm² → ~76.6% design occupancy
export DIE_AREA   = 0 0 4200 4300
export CORE_AREA  = 20 20 4180 4280
export PLACE_DENSITY = 0.55

# ── Timing ──
export CLOCK_PERIOD = 10.0
export GPL_TIMING_DRIVEN = 0

# ── Parallel jobs ──
# CRITICAL: NUM_CORES=1 to avoid parallel yosys OOM (sparsenn lesson)
export NUM_CORES = 1

# ── SRAM mock → surgery flow config ──
# For netlist surgery workflow (if needed):
#   export SYNTH_MOCK_LARGE_MEMORIES = 1
#   export SYNTH_MEMORY_MAX_BITS = 4096
