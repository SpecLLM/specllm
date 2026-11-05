# SpecLLM DSA

RTL implementation of a domain-specific accelerator for K-quant high-efficiency
forwarding in LLM speculative decoding. The DSA integrates **MsPE** (Multi-stream
Processing Elements), **QuanRedPipe** (Quantization Reduction Pipeline), and
**QuantifyEngine** to accelerate low-bit quantized MatMul with structured
reduction.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  specLLM_dsa_top                                    │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ MsPE Array│  │  FP16 Array  │  │ QuanRedPipe  │  │
│  │  (32×32)  │  │   (32×32)    │  │  (2-stage)   │  │
│  └───────────┘  └──────────────┘  └──────────────┘  │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │QuantifyEng│  │ Vector Unit  │  │DSA Controller│  │
│  │ +Buf SRAM │  │  (32-lane)   │  │   10-state   │  │
│  └───────────┘  └──────────────┘  └──────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  32 scratchpads × 16 KB + Quant Buffer 16 KB  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### MsPE — Multi-stream Processing Element

Configurable multi-stream PEs for low-bit high-throughput matrix multiplication.
Each 8-bit weight word is split into lanes — 4×2-bit for Q2K (INT2) or 2×3-bit
for Q3K (INT3) — and each lane is a multiplier-free **shift selector**, so the
lane sum equals weight × activation.

| Format | Lanes per MsPE | Throughput |
|--------|:--------------:|-----------|
| INT2×INT8 | 4 | 4 ops/cycle |
| INT3×INT8 | 2 | 2 ops/cycle |

### QuanRedPipe — Quantization Reduction Pipeline

QuanRedPipe fuses INT32→FP32 conversion and K-quant dequantization in a
two-stage pipeline.

**Stage 1.** Convert each INT32 partial sum to FP32, and precompute
$S_{\mathrm{combined}} = W_{\mathrm{scale}} \cdot W_{\mathrm{superscale}}$
together with the bias product
$W_{\mathrm{bias}} \cdot W_{\mathrm{superscale}}$.

**Stage 2.** For every channel:

$$O_j = \mathrm{FP32}(P_j) \cdot S_{\mathrm{combined}}
      + \left(W_{\mathrm{bias}} \cdot W_{\mathrm{superscale}}
      + W_{\mathrm{superbias}}\right)$$

The activation scale is folded into $W_{\mathrm{scale}}$ at the top level.

### QuantifyEngine

On-the-fly quantization engine with dedicated buffer SRAM for weight/activation
quantization and format conversion.

## Repository Structure

```
SpecLLM/
├── rtl/                          # Verilog RTL sources
│   ├── specLLM_dsa_top.v         # Top-level DSA module
│   ├── mspe_array_32x32.v        # 32×32 MsPE systolic array
│   ├── mspe_cell.v               # Single MsPE cell
│   ├── quan_red_pipe.v           # QuanRedPipe 2-stage pipeline
│   ├── quantify_engine.v         # QuantifyEngine
│   ├── quantify_buffer_sram.v    # Quantization buffer SRAM
│   ├── dsa_controller.v          # DSA control FSM
│   ├── tl_slave_if.v             # TileLink slave interface
│   ├── tl_master_dma.v           # TileLink master DMA (multi-outstanding)
│   ├── tlul_arb4.v               # 4:1 TileLink-UL round-robin arbiter
│   ├── tlul_mem_model.v          # Pipelined TileLink-UL DRAM model
│   ├── dram_model.v              # Shared DRAM model (bandwidth gate)
│   ├── vector_unit.v             # Vector ALU
│   ├── vector_reg.v              # Vector register file
│   ├── scratchpad_1024x128_freepdk45.v  # Scratchpad memory
│   ├── fp16_pe.v                 # FP16 processing element
│   ├── fp16_mul.v                # FP16 multiplier
│   ├── fp16_array_32x32.v        # 32×32 FP16 array
│   ├── fp32_add.v / fp32_mul.v   # FP32 arithmetic units
│   ├── fp32_exp.v / fp32_recip.v / fp32_sqrt.v  # FP32 transcendentals
│   ├── fp32_gelu.v               # FP32 GELU activation
│   ├── dsa_params.vh             # DSA parameter defines
│   └── sparsenn_params.vh        # SparseNN parameter defines
├── scripts/                      # Synthesis flow
│   └── synth/                    # Yosys synthesis scripts
│       ├── synth_dsa.ys
│       ├── floorplan_dsa.tcl
│       ├── gpl_quick.tcl
│       └── config_dsa.mk
├── docs/                         # Design spec
│   └── superpowers/specs/        # SoC design specification
└── README.md
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| MsPE Array | 32 × 32 |
| FP16 Array | 32 × 32 |
| Vector Unit | 32 lanes × FP32 (1024 b) |
| Supported Formats | INT2×INT8, INT3×INT8, FP16 |
| Scratchpad | 32 × (1024 × 128) = 512 KB (FreePDK45) |
| DSA die | 4200 × 4300 µm (18.06 mm²) |
| Bus Interface | TileLink |
| Process | FreePDK45 SRAM macros / Nangate45 stdcells |

## Toolchain

- **Simulation**: Verilator / Icarus Verilog
- **Synthesis**: Yosys + OpenSTA (Nangate45)
- **Floorplan**: OpenROAD
