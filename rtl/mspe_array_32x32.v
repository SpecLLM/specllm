// mspe_array_32x32.v — 32×32 MsPE Systolic Array (DynLogic, multiplier-free)
//
// Geometry: 32×32 (1024 PEs, 32 columns). Each PE weight word is 8 bit and is
// split into lanes internally (Q2K: 4×2b / Q3K: 2×3b) -- see mspe_cell.v.
//
// ── Weight port: 1024 bit ──
//   MsPE is weight-stationary: weights are latched into the PEs on `start` and
//   are not read for the rest of the 64-cycle computation. A 32×32×8b = 8192-bit
//   weight tile therefore does not need to arrive in one cycle -- a register
//   buffer filled by 8 wide SRAM reads is sufficient.
//
//   Exposing 8192 bits as module ports would have two consequences:
//     1) The hierarchical block would need 9,477 IO pins (5 + 256 + 8192 + 1024),
//        while a 1300² die perimeter only fits 9,272 pins, so `PPL-0024` fails.
//     2) The DSA top level would have to route 8192 wires across the macro
//        boundary, which is very expensive.
//   Narrowing the port to 1024 bit reduces the pin count to 2,314 and cuts
//   cross-macro wires by 87.5%.
//
// ── Weight load timing (10 cycles) ──
//   The scratchpad SRAM has a one-cycle read latency (registered dout), so the
//   address must be issued one cycle early. The array outputs `wt_load_idx`
//   combinationally for the top level to form the SRAM row address, and uses
//   `idx_d` (one cycle delayed) to select the word written by the returned
//   `wt_wide`:
//
//     cycle 0   req=1           -> busy<=1, idx<=0       (top starts driving addr=idx)
//     cycle 1   addr=0 issued    -> idx_d=0, write word 0
//     cycle 2   addr=1           -> idx_d=0, write word 0 (data of addr0) ... and so on
//     ...
//     cycle 9   last word        -> busy<=0
//
//   => The top-level controller waits 10 cycles (WT_LOAD_CYCLES = 10).
//   Steady-state occupancy is 10/64 = 15.6%; a second buffer stage could hide
//   it completely, but this version does not implement that.

module mspe_array_32x32 #(
    parameter integer N_ROWS = 32,        // rows (= activation vector width)
    parameter integer M_COLS = 32          // columns (= number of partial sums)
) (
    input  wire                                 clk,
    input  wire                                 rst_n,

    // ── Control ──
    input  wire                                 mode,        // 0 = Q2K (4 lane), 1 = Q3K (2 lane)
    input  wire                                 start,       // one cycle: latch wt_buf into all PEs
    input  wire                                 accumulate,  // 1 = accumulate across cycles, 0 = restart

    // ── Weight load (8 wide reads, 1024-bit port) ──
    input  wire                                 wt_load_req,  // 1-cycle pulse, start loading
    input  wire [1023:0]                        wt_wide,      // one 1024-bit wide read
    output reg                                  wt_load_busy, // loading in progress
    output wire [2:0]                           wt_load_idx,  // 0..7, top level forms the SRAM row address

    // ── Activation input (left edge, N_ROWS × 8b) ──
    input  wire [N_ROWS-1:0][7:0]               activations,

    // ── Partial sum output (bottom edge, M_COLS × 32b) ──
    output wire [M_COLS-1:0][31:0]              partial_sums
);

    localparam integer NBIT = N_ROWS * M_COLS * 8;      // total weight buffer bits (32×32×8 = 8192)

    // ══════════════════════════════════════════════════════════════
    // Weight register buffer + 8-cycle load FSM
    // ══════════════════════════════════════════════════════════════
    //   One weight tile = N_ROWS × M_COLS × 8b = 8192 b
    //   One wide read = 8 scratchpads × 128 b = 1024 b => 8 cycles
    reg [NBIT-1:0] wt_buf;
    reg [2:0]      idx_r;      // address index driven this cycle (combinational output)
    reg [2:0]      idx_d;      // idx_r delayed one cycle = word for the current wt_wide

    assign wt_load_idx = idx_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_load_busy <= 1'b0;
            idx_r        <= 3'd0;
            idx_d        <= 3'd0;
            wt_buf       <= {NBIT{1'b0}};
        end else if (!wt_load_busy) begin
            if (wt_load_req) begin
                wt_load_busy <= 1'b1;
                idx_r        <= 3'd0;
            end
        end else begin
            // The current wt_wide corresponds to the previous address, idx_d
            wt_buf[idx_d*1024 +: 1024] <= wt_wide;
            idx_d <= idx_r;
            if (idx_r == 3'd7) begin
                wt_load_busy <= 1'b0;              // all 8 words received
            end else begin
                idx_r <= idx_r + 3'd1;
            end
        end
    end

    // ══════════════════════════════════════════════════════════════
    // Weight distribution: PE(i,j) gets byte (i*M_COLS + j) of the tile
    //   The earlier 3-bit cross-field slicing was tied to row numbering;
    //   byte alignment removes both problems.
    // ══════════════════════════════════════════════════════════════
    wire [N_ROWS-1:0][M_COLS-1:0][7:0] mspe_weights;
    genvar wi, wj;
    generate
        for (wi = 0; wi < N_ROWS; wi = wi + 1) begin : wt_row
            for (wj = 0; wj < M_COLS; wj = wj + 1) begin : wt_col
                assign mspe_weights[wi][wj] = wt_buf[(wi*M_COLS + wj)*8 +: 8];
            end
        end
    endgenerate

    // ══════════════════════════════════════════════════════════════
    // Systolic grid
    // ══════════════════════════════════════════════════════════════
    wire [7:0]  activation_wire  [0:N_ROWS-1][0:M_COLS-1];
    wire [31:0] partial_sum_wire [0:N_ROWS-1][0:M_COLS-1];

    generate
        for (wi = 0; wi < N_ROWS; wi = wi + 1) begin : row
            for (wj = 0; wj < M_COLS; wj = wj + 1) begin : col

                // ── Activation input: array input at the left edge, left neighbour otherwise ──
                wire [7:0] act_in_wire;
                if (wj == 0)
                    assign act_in_wire = activations[wi];
                else
                    assign act_in_wire = activation_wire[wi][wj-1];

                // ── Partial-sum input: zero at the top edge (new accumulation), upper neighbour otherwise ──
                wire [31:0] psum_in_wire;
                if (wi == 0)
                    assign psum_in_wire = 32'd0;
                else
                    assign psum_in_wire = partial_sum_wire[wi-1][wj];

                // ── PE ──
                mspe_cell u_mspe (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .mode            (mode),
                    .weight_load     (start),
                    .accumulate      (accumulate),
                    .activation_in   (act_in_wire),
                    .weight          (mspe_weights[wi][wj]),
                    .partial_sum_in  (psum_in_wire),
                    .activation_out  (activation_wire[wi][wj]),
                    .partial_sum_out (partial_sum_wire[wi][wj])
                );

            end
        end

        // ── Output: bottom edge of each column ──
        // Note: the earlier candidate set {0, 2a, 4a, 8a} produced only even
        // values, so the accumulator LSB was structurally 0 and hilomap used
        // `partial_sums[j][0]` as a global constant, creating "tie cell drives
        // output port" -> OpenROAD GROUND net -> DRT-0305. Changing the
        // candidate set to {0, a, 2a, 4a} (2026-10-08) removed that structure.
        // `scripts/fix_tie_ports.py` is kept as a fallback (it reports "no
        // changes needed" and returns the input unchanged when nothing matches).
        for (wj = 0; wj < M_COLS; wj = wj + 1) begin : output_col
            assign partial_sums[wj] = partial_sum_wire[N_ROWS-1][wj];
        end
    endgenerate

endmodule
