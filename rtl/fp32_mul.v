// fp32_mul.v — IEEE 754 FP32 multiplier (single-cycle combinational)
// Reused in vector unit for softmax/layernorm scale operations.

`include "sparsenn_params.vh"

(* keep_hierarchy *) module fp32_mul (
    input  wire [`FP32_WIDTH-1:0] a,
    input  wire [`FP32_WIDTH-1:0] b,
    output reg  [`FP32_WIDTH-1:0] result
);

    wire          sign_a, sign_b;
    wire [ 7:0]   exp_a, exp_b;
    wire [22:0]   man_a, man_b;

    assign sign_a = a[31];
    assign sign_b = b[31];
    assign exp_a  = a[30:23];
    assign exp_b  = b[30:23];
    assign man_a  = a[22:0];
    assign man_b  = b[22:0];

    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan, a_is_norm, b_is_norm;

    assign a_is_zero   = (exp_a == 8'd0) && (man_a == 23'd0);
    assign a_is_denorm = (exp_a == 8'd0) && (man_a != 23'd0);
    assign a_is_inf    = (exp_a == 8'hFF) && (man_a == 23'd0);
    assign a_is_nan    = (exp_a == 8'hFF) && (man_a != 23'd0);
    assign a_is_norm   = (exp_a != 8'd0) && (exp_a != 8'hFF);

    assign b_is_zero   = (exp_b == 8'd0) && (man_b == 23'd0);
    assign b_is_denorm = (exp_b == 8'd0) && (man_b != 23'd0);
    assign b_is_inf    = (exp_b == 8'hFF) && (man_b == 23'd0);
    assign b_is_nan    = (exp_b == 8'hFF) && (man_b != 23'd0);
    assign b_is_norm   = (exp_b != 8'd0) && (exp_b != 8'hFF);

    wire [23:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a} :
                       a_is_denorm ? {1'b0, man_a} : 24'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b} :
                       b_is_denorm ? {1'b0, man_b} : 24'd0;

    wire          sign_r = sign_a ^ sign_b;
    wire [47:0]   man_product = man_a_ext * man_b_ext;
    wire [ 8:0]   exp_sum = {1'b0, exp_a} + {1'b0, exp_b};

    // Bias correction: FP32 bias = 127. Must subtract from product exponent.
    localparam BIAS = 9'd127;

    always @(*) begin
        if (a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf)) begin
            result = {1'b0, 8'hFF, 23'h400000};
        end
        else if (a_is_inf || b_is_inf) begin
            result = {sign_r, 8'hFF, 23'd0};
        end
        else if (a_is_zero || b_is_zero) begin
            result = {sign_r, 8'd0, 23'd0};
        end
        else begin
            casez (man_product)
                48'b1???????????????????????????????????????????????: begin
                    result = {sign_r, exp_sum + 9'd1 - BIAS, man_product[46:24]};
                end
                48'b01??????????????????????????????????????????????: begin
                    result = {sign_r, exp_sum - BIAS, man_product[45:23]};
                end
                48'b001?????????????????????????????????????????????: begin
                    result = {sign_r, exp_sum - 9'd1 - BIAS, man_product[44:22]};
                end
                48'b0001????????????????????????????????????????????: begin
                    result = {sign_r, exp_sum - 9'd2 - BIAS, man_product[43:21]};
                end
                default: begin
                    result = {sign_r, 8'd0, 23'd0};
                end
            endcase
        end
    end

endmodule
