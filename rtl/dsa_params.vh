// dsa_params.vh — DSA-specific parameter defines
// Extends sparsenn_params.vh with NPU-Q DSA specific parameters

`ifndef DSA_PARAMS_VH
`define DSA_PARAMS_VH

`include "sparsenn_params.vh"

// ── MsPE Array ──
`define MSPE_ROWS           64
`define MSPE_COLS           64
`define MSPE_WEIGHT_BITS      3   // 3-bit quantized weight (Q2K: 2b usable, Q3K: 3b)

// ── FP16 PE Array ──
`define FP16_PE_ROWS        32
`define FP16_PE_COLS        32

// ── QuanRedPipe ──
`define QUANRED_N_CHAN       64   // Number of parallel channels (= MSPE_COLS)

// ── Vector Unit ──
`define VECTOR_LANES         32

// ── Quantify Engine ──
`define QUANT_GROUP_SIZE    128
`define QUANT_NUM_CHANNELS   64

// ── SRAM Configuration ──
`define SRAM_DEPTH         1024   // Words per scratchpad
`define SRAM_WIDTH          128   // Bits per word
`define SRAM_BANK_COUNT       8   // Scratchpads per bank (wt/act/acc)
`define SRAM_ADDR_BITS       10   // log2(SRAM_DEPTH)

// ── TileLink Interface ──
`define TL_DATA_WIDTH        64
`define TL_ADDR_WIDTH        32

// ── DSA Controller ──
`define DSA_STATES           11

`endif // DSA_PARAMS_VH
