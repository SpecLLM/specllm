// mspe_cell.v — DynLogic Shift-Mux-Add PE, 8-bit weight input / multiple lanes
//
// ── Differences from the earlier version (important) ──
//   Earlier: `input wire [2:0] weight` -- only one lane, i.e. 1/4 (Q2K) or 1/2
//   (Q3K) of the width specified in the design document:
//       "MsPE Array (32x32) | DynLogic multiplier-free PE array.
//        Q2K->4-lane, Q3K->2-lane MAC streams"
//   This version implements that specification.
//
// ── Weight-word interpretation (8 bit, split into lanes internally) ──
//   Q2K (mode=0): 4 x 2 bit -> 4 lanes, each magnitude mag in {0,1,2,3}
//                 contributes {0, a, 2a, 4a}; the four lanes are summed.
//                 A representable weight is a sum of four values from {0,1,2,4},
//                 i.e. {0,...,16}*a.
//                 The earlier version used {0,2a,4a,8a}: 2 bits encoded 8a (2^3)
//                 and dropped 2^0, so only even weights {0,2,...,32} were possible
//                 and the accumulator LSB was always 0.
//   Q3K (mode=1): 2 x 3 bit -> 2 lanes, each 3 bit = {mag[1:0], sign}
//                 contributes +/-{0, a, 2a, 4a}; two lanes are summed
//                 (8 levels = 3 bit OK). Bit[7] is unused in Q3K.
//
//   There is still no multiplier: each lane is a shift selector (act << mag),
//   and the lane sum equals weight x activation -- the weight is encoded as a
//   sum of powers of two. Eight bits are exactly enough for 4 such terms (Q2K)
//   or 2 terms (Q3K).
//
// ── Data flow (unchanged) ──
//   weight-stationary: the 8-bit weight is latched on `weight_load` and kept
//   constant. Activations shift right every cycle; partial sums accumulate
//   downward every cycle.

module mspe_cell (
    input  wire         clk,
    input  wire         rst_n,

    // ── Control ──
    input  wire         mode,            // 0 = Q2K (4×2b), 1 = Q3K (2×3b)
    input  wire         weight_load,     // latch the 8-bit weight (weight-stationary)
    input  wire         accumulate,      // 1 = accumulate, 0 = restart

    // ── Data ──
    input  wire [7:0]   activation_in,
    input  wire [7:0]   weight,          // 8 bit per PE (earlier version used [2:0])
    input  wire [31:0]  partial_sum_in,

    output reg  [7:0]   activation_out,
    output reg  [31:0]  partial_sum_out
);

    reg  [7:0]  weight_reg;       // 8-bit resident weight
    reg  [7:0]  activation_reg;
    reg  [31:0] accum_reg;

    // ── Shift candidates (corrected set, 2026-10-08) ──
    //   Old: {0, 2a, 4a, 8a} -- 2 bits encoded exponents 1..3, dropping 2^0 = a
    //        and using 8a (= 2^3, a 3-bit quantity) in a "2-bit weight", which is
    //        inconsistent with the Q2K bit width.
    //   New: {0, a, 2a, 4a} -- what 2 bits should encode (0 + 2^0..2^2).
    //   => odd/small weights (1, 2, 3, ...) are now representable; the dynamic
    //      range shrinks from 0..8 to 0..4.
    wire [10:0] act_ext = {3'd0, activation_reg};
    wire [10:0] sh0 = 11'd0;             // 0      (mag=0)
    wire [10:0] sh1 = act_ext;           // a      (mag=1); earlier: act_ext << 1
    wire [10:0] sh2 = act_ext << 1;      // 2a     (mag=2); earlier: act_ext << 2
    wire [10:0] sh3 = act_ext << 2;      // 4a     (mag=3); earlier: act_ext << 3

    // ── One lane: 2-bit magnitude select + optional sign ──
    function signed [12:0] lane_mac;
        input [1:0] mag;
        input       sgn;
        reg   [10:0] m;
        begin
            case (mag)
                2'd0:    m = sh0;
                2'd1:    m = sh1;
                2'd2:    m = sh2;
                default: m = sh3;
            endcase
            lane_mac = sgn ? -{2'b00, m} : {2'b00, m};
        end
    endfunction

    // ── Q2K: 4 lanes, one unsigned magnitude per 2 bits ──
    wire signed [12:0] q2k0 = lane_mac(weight_reg[1:0], 1'b0);
    wire signed [12:0] q2k1 = lane_mac(weight_reg[3:2], 1'b0);
    wire signed [12:0] q2k2 = lane_mac(weight_reg[5:4], 1'b0);
    wire signed [12:0] q2k3 = lane_mac(weight_reg[7:6], 1'b0);

    // ── Q3K: 2 lanes, each 3 bits = {mag[1:0], sign} ──
    wire signed [12:0] q3k0 = lane_mac(weight_reg[1:0], weight_reg[2]);
    wire signed [12:0] q3k1 = lane_mac(weight_reg[4:3], weight_reg[5]);

    // ── Lane sum (multiplier-free approximation of weight x activation) ──
    wire signed [14:0] prod_sum = mode
        ? (q3k0 + q3k1)                                  // 2 lanes
        : (q2k0 + q2k1 + q2k2 + q2k3);                   // 4 lanes
    wire signed [31:0] product_32 = {{17{prod_sum[14]}}, prod_sum};

    // ── Systolic partial sum + temporal accumulation ──
    wire signed [31:0] psum_next  = $signed(partial_sum_in) + product_32;
    wire signed [31:0] accum_next = accumulate ? ($signed(accum_reg) + product_32)
                                               : product_32;

    // ── Pipeline registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'd0; activation_reg <= 8'd0; accum_reg <= 32'd0;
        end else begin
            if (weight_load) weight_reg <= weight;       // weight-stationary: latch once
            activation_reg <= activation_in;
            accum_reg      <= accum_next;
        end
    end

    // ── Output registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out <= 8'd0; partial_sum_out <= 32'd0;
        end else begin
            activation_out  <= activation_reg;
            partial_sum_out <= psum_next;
        end
    end

endmodule
