// dsa_params.vh — DSA-specific parameter defines
// Extends sparsenn_params.vh with NPU-Q DSA specific parameters

`ifndef DSA_PARAMS_VH
`define DSA_PARAMS_VH

`include "sparsenn_params.vh"

// ── MsPE Array ──
`define MSPE_ROWS           32
`define MSPE_COLS           32
`define MSPE_WEIGHT_BITS      8   // 8-bit weight word, split into lanes internally (Q2K: 4×2b / Q3K: 2×3b)

// ── FP16 PE Array ──
`define FP16_PE_ROWS        32
`define FP16_PE_COLS        32

// ── QuanRedPipe ──
`define QUANRED_N_CHAN       32   // Number of parallel channels (= MSPE_COLS)

// ── Vector Unit ──
`define VECTOR_LANES         32

// ── Quantify Engine ──
`define QUANT_GROUP_SIZE    128
`define QUANT_NUM_CHANNELS   16

// ── SRAM Configuration ──
`define SRAM_DEPTH         1024   // Words per scratchpad
`define SRAM_WIDTH          128   // Bits per word
`define WT_SRAM_COUNT         8   // Weight bank: 8 × 16 KB = 128 KB
`define ACT_SRAM_COUNT        8   // Activation bank: 8 × 16 KB = 128 KB
`define ACC_SRAM_COUNT       16   // Accumulator bank: 16 × 16 KB = 256 KB (2 groups × 8, ping-pong)
`define ACC_GROUP             8   // Scratchpads per group
`define SRAM_BANK_COUNT       8   // (legacy name, kept for compatibility; use the three above)
`define SRAM_ADDR_BITS       10   // log2(SRAM_DEPTH)

// ── TileLink Interface ──
`define TL_DATA_WIDTH        64
`define TL_ADDR_WIDTH        32

// ── DSA Controller ──
`define DSA_STATES           11

`endif // DSA_PARAMS_VH
