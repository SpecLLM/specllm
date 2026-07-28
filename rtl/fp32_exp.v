// fp32_exp.v — FP32 exp(x) piecewise approximation for softmax
// Simple exponent-based piecewise. Softmax uses x = data - max, so x <= 0.
// Positive x: exp(-x); negative x: exp(x).

`include "sparsenn_params.vh"

module fp32_exp (
    input  wire [`FP32_WIDTH-1:0] x,
    output reg  [`FP32_WIDTH-1:0] result
);

    wire          sign = x[31];
    wire [ 7:0]   xexp = x[30:23];

    always @(*) begin
        if (sign) begin
            // x < 0: exp(x). Piecewise by |x| range.
            // |x| < 1  → 0.5,  |x| in [1,2) → 0.25
            // |x| in [2,4) → 0.125,  |x| >= 4 → 0
            if (xexp < 8'd127)
                result = 32'h3F000000;  // 0.5
            else if (xexp == 8'd127)
                result = 32'h3E800000;  // 0.25
            else if (xexp == 8'd128)
                result = 32'h3E000000;  // 0.125
            else
                result = 32'd0;          // 0
        end else begin
            // x >= 0: exp(-x)
            // x in [0,1) → 1.0,  x in [1,2) → 0.25,  x >= 2 → 0
            if (xexp < 8'd127)
                result = 32'h3F800000;  // 1.0
            else if (xexp == 8'd127)
                result = 32'h3E800000;  // 0.25
            else
                result = 32'd0;          // 0
        end
    end

endmodule
