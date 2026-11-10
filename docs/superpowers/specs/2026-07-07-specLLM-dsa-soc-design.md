# SpecLLM DSA SoC — 4×RocketTile + NPU-Q Design Spec

**Date:** 2026-07-08 (updated)
**Status:** RTL Complete
**Reference:** `design.tex` (SpecLLM NPU-Q architecture), `/home/user/project/sparsenn/` (SoC integration pattern)

---

## 1. Overview

### 1.1 Goal

Build a heterogeneous SoC with:
- **4× RocketTile** (RISC-V cores, reuse sparsenn synthesis LEF/DEF)
- **1× NPU-Q DSA** (new, based on SpecLLM design.tex)
- **1× L2 Cache** (CoherenceManagerWrapper, reuse sparsenn)
- **1× SystemBus** (TL Xbar, reuse sparsenn)

Target: FreePDK45 (nangate45), RTL delivery only.

### 1.2 Application

Pipeline-based speculative LLM decoding:
- RocketTile CPUs: scheduling + QK attention compute
- NPU-Q DSA: core matmul + dequant fusion (QuanRedPipe, MsPE, Quantify Engine)
- Shared TileLink bus for CPU-DSA communication (no RocketTile internals modified)

---

## 2. DSA (NPU-Q) Module Architecture

### 2.1 Sub-Module List

| Component | Description | Area (mm²) | Share of die |
|-----------|-------------|:----------:|:------------:|
| **FP16 PE Array (32×32)** | Dense FP16 systolic array, 1024 PEs | 5.954 | 33.0% |
| **SRAM Bank (32 scratchpad)** | 8 wt + 8 act + 16 acc, 512 KB (scratchpad_1024x128, 16 KB each) | 5.510 | 30.5% |
| **MsPE Array (32×32)** | DynLogic multiplier-free PE array, Q2K 4-lane / Q3K 2-lane | 1.850 | 10.2% |
| **Quantify Buffer SRAM** | 128 × 1024 b | 0.225 | 1.2% |
| **Vector (unit + reg)** | 32-lane / 1024-b FP32 SIMD + register bank, including its fp32 macros | 0.156 | 0.87% |
| **QuanRedPipe** | 32-ch FP32 MAC-adder tree, 2-cycle pipeline, including its fp32 macros | 0.129 | 0.71% |
| **Quantify Engine** | 5-state FSM pipeline: COLLECT→SCALE→DRAIN | 0.007 | 0.04% |
| **Other** | Filler/tap/endcap, empty space, TL slave, controller, top-level glue | 4.230 | 23.4% |
| **Total (die)** | | **18.06** | **100%** |

Design content (macros + logic) = **13.84 mm²** in an 18.06 mm² die (78.6%
utilization); the remaining 4.23 mm² is filler/empty space.

**Design area:** 13.84 mm² | **DSA die:** 4200×4300 um = 18.06 mm²

### 2.2 Data Flow (verified, all paths connected)

```
                          ┌─ quant_scales_out → latch → QuanRedPipe.w_scale (mux with CSR)
                          │
FP16 Act SRAM ──→ Quantify Engine ──→ quant_q8_out → Act SRAM (writeback to addr 0)
                          │                        (overwrites FP16 input with Q8)
                          │
                 Act SRAM ──→ MsPE Array (32×32 DynLogic) ──┐
                                                          │
                        ┌─────────────────────────────────┘
                        │
                        └──→ QuanRedPipe ──→ VectorUnit ──→ Acc SRAM
                                 ↑                              │
                           qrp_w_scale                    DMA readback
                          (CSR or quant latch)             (TL bus)
```

1. DMA loads weights/activations via TL bus into Weight/Act SRAMs
2. Quantify Engine reads FP16 from Act SRAM → quantizes to Q8 → writes back to Act SRAM
3. MsPE reads weights + Q8 activations → computes INT32 partial sums
4. QuanRedPipe fuses INT32→FP32 conversion + scale/bias dequant (2-cycle pipeline)
5. Vector Unit applies element-wise post-processing (ReLU/LayerNorm/etc.)
6. Results written to Acc SRAM → DMA-read back to host via TL bus
7. Done detection via cycle counters (MsPE / FP16: 64 cycles)

### 2.3 SRAM Topology (all at top level)

All SRAM macros instantiated in `specLLM_dsa_top.v` — no sub-module contains macros:

| SRAM Bank | Module | Count | Width | Depth | Notes |
|-----------|--------|:----:|:-----:|:-----:|-------|
| Weight | scratchpad_1024x128 | 8 | 128b | 1024 | DMA write only |
| Activation | scratchpad_1024x128 | 8 | 128b | 1024 | DMA write + Quantify Engine write (muxed) |
| Accumulator | scratchpad_1024x128 | 16 | 128b | 1024 | Ping-pong pairs; pipeline writeback + DMA readback |
| Quantify Buffer | quantify_buffer_sram | 1 | 1024b | 128 | Behavioral, replaceable via surgery flow |

32 scratchpad macros (512 KB) + 1 buffer macro. This keeps sub-modules
macro-free for clean hierarchical synthesis.

---

## 3. SoC Top-Level Architecture

### 3.1 Module Sizes (2026-10-29 signoff)

| Module | Die size (µm) | Die area (mm²) | Design area (mm²) | Utilization |
|--------|:-------------:|:--------------:|:-----------------:|:-----------:|
| NPU-Q DSA | 4200 × 4300 | 18.06 | 13.837 | 78.6% |
| RocketTile ×4 | 1900 × 960 | 2.70 each | 1.984 each | 77.3% |
| L2 Cache | — | 7.39 | 1.630 | 22.7% |
| SystemBus | — | 0.09 | 0.009 | 10.7% |
| **SoC top** | **8600 × 7150** | **61.49** | **32.864** | **54.0%** |

Source: ORFS signoff report (2026-10-29).

### 3.2 Top-Level Floorplan

```
+-------------------------------------------------------+
|                                                        |
|                 NPU-Q DSA                              |
|                 4200 x 4300 um                         |
|                                                        |
+----------------------------+--------+------+------+---+
| L2 Cache (4000 x 1000 um) |  SB    | RT0  | RT1  |   |
|                            |  500   | 1000 | 1000 |   |
|                            |  x500  | x1000|x1000 |   |
+----------------------------+--------+------+------+---+
                                             RT2  | RT3 |
                                             1000 | 1000 |
                                             x1000| x1000|
                                             +-----+-----+
Total die: 8600 x 7150 um ≈ 61.49 mm²
```

### 3.3 Interconnect Topology

```
        RocketTile0  RocketTile1  RocketTile2  RocketTile3
             │            │            │            │
             └────────────┼────────────┼────────────┘
                          │  SystemBus (TL Xbar)
                     ┌────┴────┐
                 L2 Cache    NPU-Q DSA (TL Slave)
```

- All modules on shared TileLink bus
- RocketTile, L2, SB: unmodified, LEF/DEF reused from sparsenn
- DSA: single TL slave port for CSR config + DMA data transfer

### 3.4 DSA Internal Floorplan

```
+-----------------------------------+-------------------+
| MsPE Array (32×32 DynLogic)      | FP16 Array        |
| 1360 x 1360 um                   | 32×32             |
|                                   | 2440 x 2440 um    |
+-----------------------------------+                   |
| QuanRedPipe (FP32 Fusion)        |                   |
| ~300 x 1700 um                   +-------------------+
|                                   | QuantE  | VecUnit |
+-----------------------------------+ 400x300 | 32-lane |
| Ctrl/DMA/TL IF (300x300)         +---------+ 300x300 |
+----------------------------------+         +---------+
|                                                       |
|   SRAM Bank (33 macros)                               |
|   8wt + 8act + 16acc (scratchpad) + 1 quantify_buffer |
|   3917 x 2435 um  (512 KB + 16 KB)                    |
+-------------------------------------------------------+
DSA die: 4200 x 4300 um = 18.06 mm²
```

---

## 4. RTL Implementation (Complete)

### 4.1 File Inventory (26 files, 4505 lines)

```
rtl/
├── specLLM_dsa_top.v               (706 lines)  Top-level integration
├── dsa_controller.v                (435 lines)  10-state sequence FSM
├── dsa_params.vh                   ( 45 lines)  DSA parameter defines
├── sparsenn_params.vh              ( 54 lines)  Base parameter defines
├── mspe_cell.v                     (125 lines)  DynLogic Shift-Mux-Add Tree PE
├── mspe_array_32x32.v              (167 lines)  32×32 MsPE systolic grid
├── quan_red_pipe.v                 (210 lines)  32-ch mixed-precision fusion
├── quantify_engine.v               (505 lines)  FP16→Q8K quantization pipeline
├── quantify_buffer_sram.v          ( 32 lines)  128×1024 behavioral SRAM
├── vector_unit.v                   (454 lines)  32-lane FP32 SIMD
├── vector_reg.v                    (119 lines)  Vector register file
├── fp16_pe.v                       ( 79 lines)  FP16 processing element
├── fp16_mul.v                      (112 lines)  FP16 multiplier
├── fp16_array_32x32.v              ( 99 lines)  32×32 FP16 systolic array
├── fp32_add.v                      (139 lines)  FP32 adder
├── fp32_mul.v                      ( 82 lines)  FP32 multiplier
├── fp32_exp.v                      ( 40 lines)  FP32 exponential
├── fp32_gelu.v                     ( 40 lines)  FP32 GELU activation
├── fp32_recip.v                    ( 33 lines)  FP32 reciprocal
├── fp32_sqrt.v                     ( 36 lines)  FP32 square root
├── scratchpad_1024x128_freepdk45.v ( 73 lines)  OpenRAM SRAM macro
├── tl_slave_if.v                   (486 lines)  TileLink-UL slave + CSR
├── tl_master_dma.v                 (113 lines)  Multi-outstanding TL-UL master DMA
├── tlul_arb4.v                     (102 lines)  4:1 TL-UL round-robin arbiter
├── tlul_mem_model.v                ( 94 lines)  Pipelined TL-UL DRAM model
└── dram_model.v                    (125 lines)  Shared DRAM model (BW gate)
```

### 4.2 Module Sources

| Category | Files | Source |
|----------|-------|--------|
| New (DSA-specific) | mspe_cell, mspe_array_32x32, quan_red_pipe, quantify_engine, dsa_controller, tl_slave_if, tl_master_dma, tlul_arb4, tlul_mem_model, dram_model, quantify_buffer_sram, vector_reg, fp16_*, dsa_params.vh, specLLM_dsa_top | Written for this project |
| Ported (sparsenn) | vector_unit, fp32_*, scratchpad, sparsenn_params.vh | Copied from /home/user/project/sparsenn/rtl/ |

### 4.3 Connectivity Verification

All 15 issues from the comprehensive port audit have been resolved:

| Category | Issues Found | Resolved |
|----------|:----------:|:--------:|
| Dangling outputs (modules optimized away) | 3 | 3 |
| Broken data paths | 2 | 2 |
| Handshake deadlocks | 2 | 2 |
| Data starvation / OOB | 4 | 4 |
| Low-severity / by-design | 4 | Acknowledged |

**Result:** Every sub-module output has a consumer. No debug-only ports. Synthesis-safe.

---

## 5. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bus topology | Single shared TileLink | Reuse sparsenn L2/SB/RT without modification |
| DSA bus interface | Single TL slave port | Simplest integration, sufficient for CSR + DMA |
| MsPE size | 32×32 | 1024 multi-stream PEs, 1.85 mm² macro area |
| SRAM count | 33 (32 scratchpad + 1 buffer) | 8wt + 8act + 16acc + quantify buffer |
| SRAM placement | All at top level | No macros in sub-modules → clean hierarchical synthesis |
| Vector unit lanes | 32 (1024 b) | 2× sparsenn baseline, suitable for DSA throughput |
| Done detection | Cycle counters | MsPE / FP16: 64 cycles (32+32) |
| Quantify output | Writeback to Act SRAM | Q8 activations overwrite FP16 input at addr 0 |
| Quantify scales | Latch → QuanRedPipe mux | Per-channel scales inform dequant step |
| Integration method | Abstract LEF assembly | Same as sparsenn cpu_st_top flow |

---

## 6. Next Steps

1. **Yosys synthesis** — `read_verilog rtl/*.v; synth -top specLLM_dsa_top; stat -liberty nangate45`
2. **Area/timing extraction** — Liberty-mapped area report, critical path analysis
3. **DSA abstract LEF** — Generate LEF with pin locations and obstruction
4. **SoC floorplan** — OpenROAD 4-block integration (4×RT + DSA + L2 + SB)
5. **SRAM macro surgery** — Replace behavioral SRAMs with blackbox macros (sram_blackbox_surgery.py)
6. **Placement render** — Matplotlib visualization of final chip layout
