// fp16_mul.v — IEEE 754 FP16 multiplier producing FP32 product
// Computes: result = a * b  (both FP16, result FP32)

`include "sparsenn_params.vh"

module fp16_mul (
    input  wire [`FP16_WIDTH-1:0] a,
    input  wire [`FP16_WIDTH-1:0] b,
    output reg  [`FP32_WIDTH-1:0] result
);

    // ── Tree-structured Leading-One Detector (LOD) ──
    function [0:0] lod2;
        input [1:0] val;
        lod2 = val[1];
    endfunction
    function [1:0] lod4;
        input [3:0] val;
        lod4 = |val[3:2] ? (2'd2 + {1'b0, lod2(val[3:2])}) : {1'b0, lod2(val[1:0])};
    endfunction
    function [2:0] lod8;
        input [7:0] val;
        lod8 = |val[7:4] ? (3'd4 + {1'b0, lod4(val[7:4])}) : {1'b0, lod4(val[3:0])};
    endfunction
    function [3:0] lod16;
        input [15:0] val;
        lod16 = |val[15:8] ? (4'd8 + {1'b0, lod8(val[15:8])}) : {1'b0, lod8(val[7:0])};
    endfunction
    function [4:0] lod32;
        input [31:0] val;
        lod32 = |val[31:16] ? (5'd16 + {1'b0, lod16(val[31:16])}) : {1'b0, lod16(val[15:0])};
    endfunction

    // ── Decompose operands ──
    wire          sign_a, sign_b;
    wire [ 4:0]   exp_a, exp_b;
    wire [ 9:0]   man_a, man_b;

    assign sign_a = a[15];
    assign sign_b = b[15];
    assign exp_a  = a[14:10];
    assign exp_b  = b[14:10];
    assign man_a  = a[9:0];
    assign man_b  = b[9:0];

    // ── Classify operands ──
    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan;
    wire a_is_norm, b_is_norm;

    assign a_is_zero   = (exp_a == 5'd0) && (man_a == 10'd0);
    assign a_is_denorm = (exp_a == 5'd0) && (man_a != 10'd0);
    assign a_is_inf    = (exp_a == 5'd31) && (man_a == 10'd0);
    assign a_is_nan    = (exp_a == 5'd31) && (man_a != 10'd0);
    assign a_is_norm   = (exp_a != 5'd0) && (exp_a != 5'd31);

    assign b_is_zero   = (exp_b == 5'd0) && (man_b == 10'd0);
    assign b_is_denorm = (exp_b == 5'd0) && (man_b != 10'd0);
    assign b_is_inf    = (exp_b == 5'd31) && (man_b == 10'd0);
    assign b_is_nan    = (exp_b == 5'd31) && (man_b != 10'd0);
    assign b_is_norm   = (exp_b != 5'd0) && (exp_b != 5'd31);

    // ── Extract mantissas with hidden bits ──
    wire [10:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a} :
                       a_is_denorm ? {1'b0, man_a} : 11'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b} :
                       b_is_denorm ? {1'b0, man_b} : 11'd0;

    // ── Compute product ──
    wire          sign_r;
    wire [21:0]   man_product;     // 11 × 11 = 22 bits
    wire [ 7:0]   exp_sum_raw;     // expanded to 8 bits for FP32

    assign sign_r     = sign_a ^ sign_b;
    assign man_product = man_a_ext * man_b_ext;    // 11b × 11b = 22b
    assign exp_sum_raw = {3'd0, exp_a} + {3'd0, exp_b};

    // ── Normalization helpers (avoid part-select on expression for Yosys compat) ──
    wire [4:0]  mul_lod_pos = lod32({10'd0, man_product});
    wire [31:0] mul_man_shifted = man_product << (5'd21 - mul_lod_pos);

    // ── Exception handling (combinational) ──
    always @(*) begin
        // NaN propagation: NaN * anything = NaN
        if (a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf)) begin
            // Quiet NaN in FP32: 0_11111111_100...0
            result = {1'b0, 8'hFF, 23'h400000};
        end
        // Inf * anything non-NaN = Inf (sign depends on operands)
        else if (a_is_inf || b_is_inf) begin
            result = {sign_r, 8'hFF, 23'd0};
        end
        // Zero * anything = Zero
        else if (a_is_zero || b_is_zero) begin
            result = {sign_r, 8'd0, 23'd0};
        end
        // Denormal or normal: LOD-based normalization
        else begin
            // Zero-extend 22-bit man_product to 32 bits for uniform lod32
            if (mul_lod_pos < 5'd17) begin
                // Subnormal or very small: flush to zero
                result = {sign_r, 8'd0, 23'd0};
            end else begin
                result = {sign_r,
                          exp_sum_raw + {3'd0, mul_lod_pos} + 8'd77,
                          mul_man_shifted[20:0], 2'd0};
            end
        end
    end

endmodule
