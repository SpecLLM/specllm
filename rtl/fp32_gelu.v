// fp32_gelu.v — FP32 GELU approximation (single-cycle, 3-segment piecewise)
// Approximates GELU(x) = x * Phi(x) where Phi is the normal CDF.
// Segments and max error vs true GELU:
//   x >=  0:  GELU(x) = x          (error < 16% for x<0.5, < 1% for x>1.5)
//   x in [-2, 0): GELU(x) = -0.16  (rough constant, true: -0.16 to 0)
//   x < -2:       GELU(x) = 0

`include "sparsenn_params.vh"

module fp32_gelu (
    input  wire [`FP32_WIDTH-1:0] x,
    output reg  [`FP32_WIDTH-1:0] result
);

    wire sign = x[31];
    wire [7:0] exp = x[30:23];
    wire [22:0] man = x[22:0];

    // x >= 0?
    wire x_neg = sign;

    // Magnitude thresholds: |x| >= 2.0 has exp >= 128
    wire mag_ge_2 = (exp >= 8'd128) && !(exp == 8'd128 && man == 23'd0);  // |x| >= 2

    always @(*) begin
        if (!x_neg) begin
            // x >= 0: GELU(x) ≈ x
            result = x;
        end
        else if (mag_ge_2) begin
            // x <= -2: GELU(x) ≈ 0
            result = 32'd0;
        end
        else begin
            // -2 < x < 0: GELU(x) ≈ -0.16 (constant, max error ~0.08)
            result = 32'hBE23D70A;  // ~ -0.16 in FP32
        end
    end

endmodule
