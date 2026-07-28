// fp16_array_32x32.v — 32×32 FP16 Systolic Array
// Weight-stationary dataflow:
//   1. Preload weights column-by-column from top edge
//   2. Stream activations from left edge, propagate rightward each cycle
//   3. Accumulate partial sums downward
//   4. Drain partial sums from bottom edge
//
// Parameters: N_ROWS × M_COLS (default 32×32)
// Data format: FP16 activation × FP16 weight → FP32 accumulation

`include "dsa_params.vh"

module fp16_array_32x32 #(
    parameter N_ROWS = `FP16_PE_ROWS,     // Number of PE rows (activation dimension)
    parameter M_COLS = `FP16_PE_COLS      // Number of PE columns (weight/output dimension)
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // ── Control ──
    input  wire                         weight_load,       // Enable weight preload
    input  wire [M_COLS-1:0]            weight_col_sel,    // One-hot column select during preload
    input  wire                         accum_clear,       // Clear all accumulators
    input  wire                         stall,             // Stall all PEs

    // ── Activation input (left edge, N_ROWS × 16b) ──
    input  wire [N_ROWS*`FP16_WIDTH-1:0]  activations_in,

    // ── Weight input (top edge, N_ROWS × 16b per column during preload) ──
    input  wire [N_ROWS*`FP16_WIDTH-1:0]  weights_in,

    // ── Partial sum input (top edge, M_COLS × 32b) ──
    input  wire [M_COLS*`FP32_WIDTH-1:0]  partial_sums_in,

    // ── Partial sum output (bottom edge, M_COLS × 32b) ──
    output wire [M_COLS*`FP32_WIDTH-1:0]  partial_sums_out
);

    genvar i, j;

    // ── Wire grid for inter-PE connections ──
    // activation_wire[i][j]: activation flowing RIGHT from PE[row=i][col=j]
    // partial_sum_wire[i][j]: partial sum flowing DOWN from PE[row=i][col=j]
    wire [`FP16_WIDTH-1:0] activation_wire  [0:N_ROWS-1][0:M_COLS];
    wire [`FP32_WIDTH-1:0] partial_sum_wire [0:N_ROWS][0:M_COLS-1];

    generate
        for (i = 0; i < N_ROWS; i = i + 1) begin : row
            for (j = 0; j < M_COLS; j = j + 1) begin : col

                // ── Activation input mux ──
                // Left edge: from array input; internal: from left neighbor
                wire [`FP16_WIDTH-1:0] act_in_wire;
                if (j == 0)
                    assign act_in_wire = activations_in[i*`FP16_WIDTH +: `FP16_WIDTH];
                else
                    assign act_in_wire = activation_wire[i][j-1];

                // ── Weight input mux ──
                // All rows in column j get the same weight during preload
                wire [`FP16_WIDTH-1:0] wgt_in_wire;
                assign wgt_in_wire = weights_in[i*`FP16_WIDTH +: `FP16_WIDTH];

                // ── Partial sum input mux ──
                // Top edge: from array input; internal: from top neighbor
                wire [`FP32_WIDTH-1:0] psum_in_wire;
                if (i == 0)
                    assign psum_in_wire = partial_sums_in[j*`FP32_WIDTH +: `FP32_WIDTH];
                else
                    assign psum_in_wire = partial_sum_wire[i-1][j];

                // ── PE instantiation ──
                wire pe_weight_load;
                assign pe_weight_load = weight_load && weight_col_sel[j];

                fp16_pe u_pe (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .weight_load     (pe_weight_load),
                    .accum_clear     (accum_clear),
                    .stall           (stall),
                    .activation_in   (act_in_wire),
                    .weight_in       (wgt_in_wire),
                    .partial_sum_in  (psum_in_wire),
                    .activation_out  (activation_wire[i][j]),
                    .partial_sum_out (partial_sum_wire[i][j])
                );

            end
        end

        // ── Output connections ──
        for (j = 0; j < M_COLS; j = j + 1) begin : output_col
            assign partial_sums_out[j*`FP32_WIDTH +: `FP32_WIDTH] = partial_sum_wire[N_ROWS-1][j];
        end

    endgenerate

endmodule
