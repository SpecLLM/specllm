// fp32_sqrt.v — FP32 square root, single-cycle combinational
// Uses integer bit-manipulation magic constant.
// sqrt(x) ≈ reinterpret_as_float(MAGIC + (reinterpret_as_int(x) >> 1))
// MAGIC = 0x1FC00000 calibrated for FP32 bias=127.
// Reference: Gemmini IntSqrt (16-cycle digit-by-digit for exact results).

`include "sparsenn_params.vh"

module fp32_sqrt (
    input  wire [`FP32_WIDTH-1:0] x,
    output reg  [`FP32_WIDTH-1:0] result
);

    wire sign   = x[31];
    wire [7:0]  exp = x[30:23];
    wire [22:0] man = x[22:0];

    wire x_zero   = (exp == 8'd0) && (man == 23'd0);
    wire x_inf    = (exp == 8'hFF) && (man == 23'd0);
    wire x_nan    = (exp == 8'hFF) && (man != 23'd0);
    wire x_neg    = sign && !x_zero;

    // sqrt(x) ≈ (x_as_int >> 1) + 0x1FC00000
    // Calibration: sqrt(1.0) → 0x1FC00000 + (0x3F800000>>1) = 0x1FC00000 + 0x1FC00000 = 0x3F800000 = 1.0 ✓
    wire [31:0] x_int  = {sign, exp, man};
    wire [31:0] guess  = 32'h1FC00000 + {1'b0, x_int[31:1]};

    always @(*) begin
        if (x_zero)       result = 32'd0;
        else if (x_neg)   result = 32'h7FC00000;   // NaN
        else if (x_inf)   result = 32'h7F800000;   // inf
        else if (x_nan)   result = 32'h7FC00000;   // NaN
        else              result = guess;
    end

endmodule
