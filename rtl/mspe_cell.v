// mspe_cell.v — Multi-stream Processing Element: Shift-Mux-Add Tree MAC
// Multiplier-free PE based on DynLogic — replaces physical multiplier with
// a Shift-Mux-Add tree controlled by the quantized weight.
//
// Q2K mode (mode=0): 2-bit weight → 4-way MUX on {0, act<<1, act<<2, act<<3}
//   weight[1:0] = magnitude select, weight[2] = sign bit
// Q3K mode (mode=1): 3-bit weight → 8-way MUX on signed shifted values
//   weight[2:0] selects from {+0, +2a, +4a, +8a, -0, -2a, -4a, -8a}
//
// Weight-stationary: weight latched on weight_load, activation streams each cycle.
// Accumulator register enables multi-cycle temporal dot-product accumulation.
// Systolic dataflow: activation propagates left-to-right, partial sum top-to-bottom.

module mspe_cell (
    input  wire         clk,
    input  wire         rst_n,

    // ── Control ──
    input  wire         mode,            // 0=Q2K (2-bit weight), 1=Q3K (3-bit weight)
    input  wire         weight_load,     // Latch new stationary weight
    input  wire         accumulate,      // 1=add to running accumulator, 0=restart accumulation

    // ── Data ──
    input  wire [7:0]   activation_in,   // INT8 activation from left neighbor
    input  wire [2:0]   weight,          // Quantized weight (2-bit or 3-bit)
    input  wire [31:0]  partial_sum_in,  // INT32 partial sum from top neighbor

    output reg  [7:0]   activation_out,  // To right neighbor (registered)
    output reg  [31:0]  partial_sum_out  // To bottom neighbor (registered)
);

    // ── Internal registers ──
    reg  [2:0]  weight_reg;       // Stationary quantized weight
    reg  [7:0]  activation_reg;   // Pipelined activation
    reg  [31:0] accum_reg;        // Temporal accumulation register (INT32)

    // ── Shifted activation values for DynLogic tree ──
    // activation is zero-extended to 11 bits to accommodate act << 3 (max shift = 3)
    // sh0 = 0, sh1 = act*2, sh2 = act*4, sh3 = act*8
    wire [10:0] act_ext;
    wire [10:0] act_sh0, act_sh1, act_sh2, act_sh3;

    assign act_ext = {3'd0, activation_reg};

    assign act_sh0 = 11'd0;              // 0 * act
    assign act_sh1 = act_ext << 1;       // 2 * act
    assign act_sh2 = act_ext << 2;       // 4 * act
    assign act_sh3 = act_ext << 3;       // 8 * act

    // ── Shift-Mux Tree: magnitude select ──
    // Q2K: weight[1:0] drives 4-way MUX on {act_sh0, act_sh1, act_sh2, act_sh3}
    // Q3K: weight[2:0] drives 8-way MUX covering positive and negative values
    reg  [10:0] mux_magnitude;  // Unsigned magnitude from MUX tree
    reg         mux_sign;       // Sign from weight MSB (1 = negative)

    always @(*) begin
        if (!mode) begin
            // ── Q2K mode: 4-way MUX on weight[1:0], sign from weight[2] ──
            case (weight_reg[1:0])
                2'd0:    mux_magnitude = act_sh0;
                2'd1:    mux_magnitude = act_sh1;
                2'd2:    mux_magnitude = act_sh2;
                2'd3:    mux_magnitude = act_sh3;
                default: mux_magnitude = act_sh0;
            endcase
            mux_sign = weight_reg[2];
        end else begin
            // ── Q3K mode: 8-way MUX on weight[2:0] ──
            // Symmetric quantization: MSB encodes sign,
            // lower 2 bits encode magnitude from {act_sh0, act_sh1, act_sh2, act_sh3}
            case (weight_reg[2:0])
                3'd0:    begin mux_magnitude = act_sh0; mux_sign = 1'b0; end  // +0
                3'd1:    begin mux_magnitude = act_sh1; mux_sign = 1'b0; end  // +2a
                3'd2:    begin mux_magnitude = act_sh2; mux_sign = 1'b0; end  // +4a
                3'd3:    begin mux_magnitude = act_sh3; mux_sign = 1'b0; end  // +8a
                3'd4:    begin mux_magnitude = act_sh0; mux_sign = 1'b1; end  // -0
                3'd5:    begin mux_magnitude = act_sh1; mux_sign = 1'b1; end  // -2a
                3'd6:    begin mux_magnitude = act_sh2; mux_sign = 1'b1; end  // -4a
                3'd7:    begin mux_magnitude = act_sh3; mux_sign = 1'b1; end  // -8a
                default: begin mux_magnitude = act_sh0; mux_sign = 1'b0; end
            endcase
        end
    end

    // ── Sign handling: produce 12-bit signed product ──
    wire [11:0] signed_product;
    assign signed_product = mux_sign ? -{1'b0, mux_magnitude}
                                     :  {1'b0, mux_magnitude};

    // Sign-extend to 32 bits for INT32 accumulation
    wire [31:0] product_32;
    assign product_32 = {{20{signed_product[11]}}, signed_product};

    // ── Systolic partial sum: psum_in + product ──
    wire [31:0] psum_next;
    assign psum_next = $signed(partial_sum_in) + $signed(product_32);

    // ── Temporal accumulator: running dot product across cycles ──
    wire [31:0] accum_next;
    assign accum_next = accumulate ? ($signed(accum_reg) + $signed(product_32))
                                    : product_32;

    // ── Pipeline registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= 3'd0;
            activation_reg <= 8'd0;
            accum_reg      <= 32'd0;
        end else begin
            if (weight_load)
                weight_reg <= weight;

            activation_reg <= activation_in;
            accum_reg      <= accum_next;
        end
    end

    // ── Output registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out  <= 8'd0;
            partial_sum_out <= 32'd0;
        end else begin
            activation_out  <= activation_reg;
            partial_sum_out <= psum_next;
        end
    end

endmodule
