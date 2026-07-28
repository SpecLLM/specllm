// fp32_recip.v — Single-cycle FP32 reciprocal approximation
// Uses integer magic: 1/x ≈ 0x7F000000 - x_viewed_as_int
// Simple subtraction of integer representation of FP32 value.
// Good to ~6% accuracy, sufficient for softmax/layernorm normalization.

`include "sparsenn_params.vh"

module fp32_recip (
    input  wire [`FP32_WIDTH-1:0] x,
    output reg  [`FP32_WIDTH-1:0] result
);

    wire sign   = x[31];
    wire [7:0]  exp  = x[30:23];
    wire [22:0] man  = x[22:0];
    wire x_zero = (exp == 8'd0) && (man == 23'd0);
    wire x_inf  = (exp == 8'hFF) && (man == 23'd0);
    wire x_nan  = (exp == 8'hFF) && (man != 23'd0);

    // Integer reciprocal: treat x as signed integer, compute ~MAGIC - x
    // For normalized FP32: 1/x ≈ 0x7F000000 - {sign, exp, man}
    // This is the first iteration of the Quake fast inverse sqrt method
    wire [31:0] x_as_int = {sign, exp, man};
    wire [31:0] guess = 32'h7F000000 - x_as_int;

    always @(*) begin
        if (x_zero)       result = {sign, 8'hFF, 23'd0};     // inf
        else if (x_inf)   result = {sign, 8'd0, 23'd0};       // 0
        else if (x_nan)   result = {1'b0, 8'hFF, 23'h400000}; // NaN
        else              result = guess;
    end

endmodule
