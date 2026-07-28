// fp32_add.v — IEEE 754 FP32 adder with normalization and rounding

`include "sparsenn_params.vh"

(* keep_hierarchy *) module fp32_add (
    input  wire [`FP32_WIDTH-1:0] a,
    input  wire [`FP32_WIDTH-1:0] b,
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

    // ── Decompose ──
    wire          sign_a, sign_b;
    wire [ 7:0]   exp_a, exp_b;
    wire [22:0]   man_a, man_b;

    assign sign_a = a[31];
    assign sign_b = b[31];
    assign exp_a  = a[30:23];
    assign exp_b  = b[30:23];
    assign man_a  = a[22:0];
    assign man_b  = b[22:0];

    // ── Classify ──
    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan,  a_is_norm, b_is_norm;

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

    // ── Extended mantissas (24 bits with hidden bit, plus guard=2, sticky=1 for subnormal detection → 27 bits total) ──
    wire [26:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a, 3'd0} :
                       a_is_denorm ? {1'b0, man_a, 3'd0} : 27'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b, 3'd0} :
                       b_is_denorm ? {1'b0, man_b, 3'd0} : 27'd0;

    // ── Alignment and add/subtract ──
    wire [7:0] exp_diff, exp_large, exp_small;
    wire       sign_large, sign_small;
    wire       effective_sub;  // Are we actually subtracting?
    wire [26:0] man_large, man_small;
    wire [26:0] man_large_final, man_small_shifted;
    wire [27:0] man_sum_raw;

    // Determine which is larger (or tie-break by mantissa)
    wire a_larger = (exp_a > exp_b) || ((exp_a == exp_b) && (man_a_ext >= man_b_ext));

    assign exp_large  = a_larger ? exp_a : exp_b;
    assign sign_large = a_larger ? sign_a : sign_b;
    assign man_large  = a_larger ? man_a_ext : man_b_ext;

    assign exp_small  = a_larger ? exp_b : exp_a;
    assign sign_small = a_larger ? sign_b : sign_a;
    assign man_small  = a_larger ? man_b_ext : man_a_ext;

    assign exp_diff = exp_large - exp_small;

    // Shift small mantissa right by exp_diff
    assign man_small_shifted = (exp_diff >= 8'd27) ? 27'd0 : (man_small >> exp_diff);

    assign effective_sub = sign_large ^ sign_small;
    assign man_large_final = man_large;

    // Add or subtract
    assign man_sum_raw = effective_sub ?
        {1'b0, man_large_final} - {1'b0, man_small_shifted} :
        {1'b0, man_large_final} + {1'b0, man_small_shifted};

    // ── Normalization helpers (avoid part-select on expression for Yosys compat) ──
    wire [4:0]  add_lod_pos = lod32({4'd0, man_sum_raw});
    wire [31:0] add_man_shifted = man_sum_raw << (5'd27 - add_lod_pos);

    // ── Result interpretation and normalization ──
    always @(*) begin
        // NaN check
        if (a_is_nan || b_is_nan) begin
            result = {1'b0, 8'hFF, 23'h400000};  // Quiet NaN
        end
        // Inf + (-Inf) = NaN
        else if (a_is_inf && b_is_inf && (sign_a != sign_b)) begin
            result = {1'b0, 8'hFF, 23'h400000};  // NaN
        end
        // Inf + anything = Inf
        else if (a_is_inf) begin
            result = {sign_a, 8'hFF, 23'd0};
        end
        else if (b_is_inf) begin
            result = {sign_b, 8'hFF, 23'd0};
        end
        // Exact zero result (including cancellation)
        else if (man_sum_raw == 28'd0) begin
            result = {1'b0, 8'd0, 23'd0};  // +0 (IEEE 754 says +0 for sum of opposite signs)
        end
        else begin
            // LOD-based normalization: find leading-1, barrel-shift mantissa
            // Zero-extend 28-bit man_sum_raw to 32 bits for uniform lod32
            if (add_lod_pos < 5'd19) begin
                // Very small: underflow to zero (flush denormal)
                result = {sign_large, 8'd0, 23'd0};
            end else begin
                result = {sign_large,
                          exp_large + {3'd0, add_lod_pos} - 8'd26,
                          add_man_shifted[26:4]};
            end
        end
    end

endmodule
