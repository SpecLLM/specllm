// fp16_pe.v — Single Processing Element: FP16 multiply + FP32 accumulate
// Weight-stationary: weight preloaded once, activation streams every cycle.
// Outputs activation to right, partial sum to bottom (registered).
// Uses fp16_mul (FP16×FP16→FP32) and fp32_add for accumulation.

`include "dsa_params.vh"

module fp16_pe (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── Control ──
    input  wire                     weight_load,     // Latch new stationary weight
    input  wire                     accum_clear,     // Clear accumulation register
    input  wire                     stall,           // Stall pipeline

    // ── Data ──
    input  wire [`FP16_WIDTH-1:0]   activation_in,   // FP16 from left neighbor
    input  wire [`FP16_WIDTH-1:0]   weight_in,       // FP16 weight (latched on weight_load)
    input  wire [`FP32_WIDTH-1:0]   partial_sum_in,  // FP32 from top neighbor

    output reg  [`FP16_WIDTH-1:0]   activation_out,  // To right neighbor (registered)
    output reg  [`FP32_WIDTH-1:0]   partial_sum_out  // To bottom neighbor (registered)
);

    // ── Internal registers ──
    reg [`FP16_WIDTH-1:0]   weight_reg;
    reg [`FP16_WIDTH-1:0]   activation_reg;
    reg [`FP32_WIDTH-1:0]   accum_reg;

    // ── Arithmetic ──
    wire [`FP32_WIDTH-1:0]  mul_result;   // FP16×FP16 → FP32 product
    wire [`FP32_WIDTH-1:0]  add_result;   // FP32 sum

    // FP16 multiply: 16b × 16b → 32b (FP32 product)
    fp16_mul u_fp16_mul (
        .a      (weight_reg),
        .b      (activation_reg),
        .result (mul_result)
    );

    // FP32 accumulate: partial_sum_in + mul_result
    fp32_add u_fp32_add (
        .a      (partial_sum_in),
        .b      (mul_result),
        .result (add_result)
    );

    // ── Registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= {`FP16_WIDTH{1'b0}};
            activation_reg <= {`FP16_WIDTH{1'b0}};
            accum_reg      <= {`FP32_WIDTH{1'b0}};
        end else if (!stall) begin
            if (weight_load)
                weight_reg <= weight_in;

            activation_reg <= activation_in;

            if (accum_clear)
                accum_reg <= {`FP32_WIDTH{1'b0}};
            else
                accum_reg <= add_result;
        end
    end

    // ── Pipeline outputs ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out  <= {`FP16_WIDTH{1'b0}};
            partial_sum_out <= {`FP32_WIDTH{1'b0}};
        end else if (!stall) begin
            activation_out  <= activation_reg;
            partial_sum_out <= accum_clear ? partial_sum_in : add_result;
        end
    end

endmodule
