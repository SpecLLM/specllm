// mspe_array_64x64.v — 64x64 MsPE Systolic Array
// Multiplier-free systolic array based on DynLogic MsPE cells.
// Weight-stationary dataflow:
//   1. Load weights into all PEs on start pulse (one-shot weight preload)
//   2. Stream activations from left edge, propagate rightward each cycle
//   3. Accumulate partial sums downward through the column
//   4. Drain partial sums from bottom edge
//
// Geometry: 64 rows x 64 columns
// Q2K mode (mode=0): 2-bit weights, 4-way MUX per PE
// Q3K mode (mode=1): 3-bit weights, 8-way MUX per PE
// Data format: INT8 activation × quantized weight → INT32 accumulation

module mspe_array_64x64 (
    input  wire                         clk,
    input  wire                         rst_n,

    // ── Control ──
    input  wire                         mode,            // 0=Q2K (2-bit), 1=Q3K (3-bit)
    input  wire                         start,           // Pulse to latch weights into all PEs
    input  wire                         accumulate,      // 1=accumulate over cycles, 0=restart

    // ── Activation input (left edge, 64 rows × 8b = 512b) ──
    input  wire [63:0][7:0]             activations,

    // ── Weight input (64×64 grid × 3b = 12288b) ──
    // weights[i][j] = weight for PE at row i, column j
    input  wire [63:0][63:0][2:0]       weights,

    // ── Partial sum output (bottom edge, 64 columns × 32b = 2048b) ──
    output wire [63:0][31:0]            partial_sums
);

    // ── Parameters ──
    localparam N_ROWS = 64;
    localparam M_COLS = 64;

    genvar i, j;

    // ── Wire grid for inter-PE connections ──
    // activation_wire[i][j]: activation flowing RIGHT from PE[row=i][col=j]
    // partial_sum_wire[i][j]: partial sum flowing DOWN from PE[row=i][col=j]
    wire [7:0]  activation_wire  [0:N_ROWS-1][0:M_COLS-1];
    wire [31:0] partial_sum_wire [0:N_ROWS-1][0:M_COLS-1];

    generate
        for (i = 0; i < N_ROWS; i = i + 1) begin : row
            for (j = 0; j < M_COLS; j = j + 1) begin : col

                // ── Activation input mux ──
                // Left edge (col 0): from array input; internal: from left neighbor
                wire [7:0] act_in_wire;
                if (j == 0)
                    assign act_in_wire = activations[i];
                else
                    assign act_in_wire = activation_wire[i][j-1];

                // ── Weight input ──
                // Each PE gets its stationary weight directly from the weight grid
                wire [2:0] wgt_in_wire;
                assign wgt_in_wire = weights[i][j];

                // ── Partial sum input mux ──
                // Top edge (row 0): zero (fresh accumulation); internal: from top neighbor
                wire [31:0] psum_in_wire;
                if (i == 0)
                    assign psum_in_wire = 32'd0;
                else
                    assign psum_in_wire = partial_sum_wire[i-1][j];

                // ── MsPE instantiation ──
                mspe_cell u_mspe (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .mode            (mode),
                    .weight_load     (start),
                    .accumulate      (accumulate),
                    .activation_in   (act_in_wire),
                    .weight          (wgt_in_wire),
                    .partial_sum_in  (psum_in_wire),
                    .activation_out  (activation_wire[i][j]),
                    .partial_sum_out (partial_sum_wire[i][j])
                );

            end
        end

        // ── Output connections: bottom edge of each column ──
        for (j = 0; j < M_COLS; j = j + 1) begin : output_col
            assign partial_sums[j] = partial_sum_wire[N_ROWS-1][j];
        end

    endgenerate

endmodule
