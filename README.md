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
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ MsPE Array│  │ QuanRedPipe  │  │QuantifyEngine│ │
│  │  (64×64)  │  │  (3-stage)   │  │ + Buffer SRAM│ │
│  └───────────┘  └──────────────┘  └──────────────┘ │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │Vector Unit│  │DSA Controller│  │ Scratchpad   │ │
│  └───────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

### MsPE — Multi-stream Processing Element

Configurable multi-stream PEs for low-bit high-throughput matrix multiplication.
Each MsPE uses a **transmission gate (TG)** to gate weights and a
**Shift-Add-Tree** to simplify complex INT8 MAC operations, fully utilizing the
available bus width.

| Format | Throughput per MsPE | Density vs INT8×INT8 |
|--------|---------------------|----------------------|
| INT2×INT8 | 4 ops/cycle | 4× |
| INT3×INT8 | 2 ops/cycle | 2× |

Each stream carries an accumulator tag and group counter. All formats converge
at the `GroupReady` signal. Non-native formats fall back to the baseline vector
path and are reported separately.

### QuanRedPipe — Quantization Reduction Pipeline

QuanRedPipe preserves the K-Quant hierarchy across three logical stages per
super-group.

**Stage 1 — Group-weighted partials.** Consume completed integer results
$P_{j,g}$ and $R_{j,g}$, then apply group-level weight parameters
$W_{\mathrm{scale}}^{s,g}$ and $W_{\mathrm{bias}}^{s,g}$:

$$U_{j,s} = \sum_{g \in \mathcal{G}(s)} W_{\mathrm{scale}}^{s,g} \, P_{j,g},
\qquad
V_{j,s} = \sum_{g \in \mathcal{G}(s)} W_{\mathrm{bias}}^{s,g} \, R_{j,g}$$

**Stage 2 — Super-group reconstruction.** Reconstruct the floating-point
contribution using super-group parameters
$W_{\mathrm{superscale}}^{s}$ and $W_{\mathrm{superbias}}^{s}$:

$$Z_{j,s} = W_{\mathrm{superscale}}^{s} \, U_{j,s}
          + W_{\mathrm{superbias}}^{s} \, V_{j,s}$$

The sign convention required by a particular quantization format is absorbed
into the corresponding bias parameters. Symmetric formats gate the $V_{j,s}$
and $W_{\mathrm{superbias}}^{s}$ path.

**Stage 3 — Activation-scaled accumulation.** Apply the activation scale
$A_{\mathrm{scale}}$ and accumulate:

$$O_j \leftarrow O_j + A_{\mathrm{scale}} \, Z_{j,s}$$

### QuantifyEngine

On-the-fly quantization engine with dedicated buffer SRAM for weight/activation
quantization and format conversion.

## Repository Structure

```
SpecLLM/
├── rtl/                          # Verilog RTL sources
│   ├── specLLM_dsa_top.v         # Top-level DSA module
│   ├── mspe_array_64x64.v        # 64×64 MsPE systolic array
│   ├── mspe_cell.v               # Single MsPE cell
│   ├── quan_red_pipe.v           # QuanRedPipe 3-stage pipeline
│   ├── quantify_engine.v         # QuantifyEngine
│   ├── quantify_buffer_sram.v    # Quantization buffer SRAM
│   ├── dsa_controller.v          # DSA control FSM
│   ├── tl_slave_if.v             # TileLink slave interface
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
├── scripts/                      # Visualization & synthesis
│   ├── synth/                    # Yosys synthesis scripts
│   │   ├── synth_dsa.ys
│   │   ├── floorplan_dsa.tcl
│   │   ├── gpl_quick.tcl
│   │   └── config_dsa.mk
│   └── render_*.py               # Placement/heatmap render scripts
├── results/                      # Layout & GDS results
├── docs/                         # Design specs & progress logs
│   └── superpowers/specs/        # SoC design specification
└── README.md
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| MsPE Array | 64 × 64 |
| FP16 Array | 32 × 32 |
| Supported Formats | INT2×INT8, INT3×INT8, FP16 |
| Scratchpad | 1024 × 128 (FreePDK45) |
| Bus Interface | TileLink |
| Process | FreePDK45 |

## Toolchain

- **Simulation**: Verilator / Icarus Verilog
- **Synthesis**: Yosys + OpenSTA (FreePDK45)
- **Floorplan**: OpenROAD
- **Visualization**: Python (matplotlib)
