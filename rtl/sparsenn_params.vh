// sparsenn_params.vh — Global parameter defines for SparseNN

`ifndef SPARSENN_PARAMS_VH
`define SPARSENN_PARAMS_VH

// ── Systolic Array Geometry ──
`define N_PE_ROWS          16      // Number of PE rows
`define M_PE_COLS          16      // Number of PE columns (N,M independently configurable)

// ── Data Types (IEEE 754) ──
`define FP16_WIDTH          16
`define FP32_WIDTH          32
`define FP16_EXP_BITS        5
`define FP16_MAN_BITS       10
`define FP32_EXP_BITS        8
`define FP32_MAN_BITS       23
`define FP16_BIAS           15
`define FP32_BIAS          127

// ── Memory Sizing ──
`define WEIGHT_BUF_DEPTH  4096     // Weight scratchpad rows (each row = M_PE_COLS * FP16_WIDTH bits)
                                   // Total = 4096 * 16 * 16 = 128 KB
`define ACT_BUF_DEPTH     4096     // Activation scratchpad rows
`define ACC_BUF_DEPTH     1024     // Accumulator bank rows (each row = N_PE_ROWS * FP32_WIDTH bits)
                                   // Total = 1024 * 16 * 32 = 64 KB

// ── Sparsity Parameters (2:4 structured) ──
`define SPARSITY_N           2     // Non-zeros per group
`define SPARSITY_M           4     // Group size
`define SPARSITY_META_BITS   3     // ceil(log2(C(M,N))) = ceil(log2(6)) = 3

// ── AXI4-Stream Interface ──
`define AXI_DATA_WIDTH     256     // 16 FP16 values per beat
`define AXI_KEEP_WIDTH       32
`define AXI_ID_WIDTH          4
`define AXI_ADDR_WIDTH       32

// ── Control ──
`define MAX_TILE_ITERATIONS  65536
`define CONTROLLER_STATES       12

// ── Pipeline Options ──
`define PIPELINE_PE            1    // 0=single-cycle PE, 1=2-stage pipeline

// ── SRAM Buffers for Layout ──
// 0 = DFF-based behavioral buffers (simulation/verification)
// 1 = OpenRAM SRAM macros (physical layout)
`define USE_SRAM               1

// ── FP16T3 Ternary Core ──
`define TERN_WT_WIDTH          2
`define WBUF_FP16T3_WIDTH    512   // N*M*TERN_WT_WIDTH = 16*16*2

`endif // SPARSENN_PARAMS_VH
