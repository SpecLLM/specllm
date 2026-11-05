// quan_red_pipe.v — N-channel mixed-precision fusion pipeline (default 16)
// Pipe 2 of QuanRedPipe: intercepts INT32 partial sums and applies
// block-level scaling (w_scale / w_bias) and superblock-level adjustments
// (w_sup_scale / w_sup_bias) in a 2-cycle pipeline.
//
// Clever pre-computation: broadcast w_scale, w_bias, w_sup_scale, w_sup_bias
// are combined into a single effective scale+bias pair in Stage 1, then each
// of the N_CHAN channels needs only one fp32_mul + one fp32_add in Stage 2.
// This saves 2×N_CHAN FP ops per cycle vs. the naive 4-op-per-channel approach.

`include "sparsenn_params.vh"

module quan_red_pipe #(
    parameter N_CHAN = 64
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // ── Data inputs ──
    input  wire [N_CHAN-1:0][`FP32_WIDTH-1:0] partial_sums_in,  // N_CHAN INT32 partial sums (stored as FP32-sized words)
    input  wire                      valid_in,

    // ── Block-level parameters (broadcast, change per block) ──
    input  wire [`FP32_WIDTH-1:0]    w_scale,
    input  wire [`FP32_WIDTH-1:0]    w_bias,

    // ── Superblock-level parameters (broadcast, change less frequently) ──
    input  wire [`FP32_WIDTH-1:0]    w_sup_scale,
    input  wire [`FP32_WIDTH-1:0]    w_sup_bias,

    // ── Block index within superblock ──
    input  wire [7:0]                channel_idx,

    // ── Data outputs ──
    output reg  [N_CHAN-1:0][`FP32_WIDTH-1:0] results_out,
    output reg                       valid_out
);

    // ═══════════════════════════════════════════════════════════════
    // INT32 → FP32 conversion function
    // Handles: zero (→ +0), positive, negative, INT32_MIN safely.
    // Purely combinational.
    // ═══════════════════════════════════════════════════════════════

    function [`FP32_WIDTH-1:0] int32_to_fp32;
        input [`FP32_WIDTH-1:0] int_val;
        reg [`FP32_WIDTH-1:0] abs_val;
        reg [4:0] lead_pos;
        reg       found;
        integer   i;
        reg [54:0] abs_val_shifted;
        begin
            if (int_val == 32'd0) begin
                int32_to_fp32 = 32'd0;
            end else begin
                // Sign
                int32_to_fp32[31] = int_val[31];
                // Absolute value (two's complement)
                abs_val = int_val[31] ? (~int_val + 1'b1) : int_val;

                // Leading-one detector: priority encoder on abs_val
                found = 1'b0;
                lead_pos = 5'd0;
                for (i = 31; i >= 0; i = i - 1) begin
                    if (!found && abs_val[i]) begin
                        lead_pos = i[4:0];
                        found = 1'b1;
                    end
                end

                // Exponent: bias + position of MSB
                int32_to_fp32[30:23] = 8'd127 + {3'd0, lead_pos};

                // Mantissa: 23 bits following the implicit leading 1
                // Method: left-shift abs_val so the leading 1 lands at bit 23,
                // then bits [22:0] are the mantissa. For lead_pos >= 23,
                // we take the slice directly; for lead_pos < 23 we shift.
                // Use intermediate wire to avoid part-select on expression
                // (Yosys read_verilog -sv compatibility, ref: sparsenn synth.md)
                abs_val_shifted = abs_val << (5'd23 - lead_pos);
                if (lead_pos >= 5'd23) begin
                    int32_to_fp32[22:0] = abs_val[lead_pos-1 -: 23];
                end else begin
                    int32_to_fp32[22:0] = abs_val_shifted[22:0];
                end
            end
        end
    endfunction

    // ═══════════════════════════════════════════════════════════════
    // Stage 1: combinatorial logic
    //   (a) INT32 → FP32 for all N_CHAN channels
    //   (b) Pre-compute combined_scale  = w_scale * w_sup_scale
    //   (c) Pre-compute w_bias_sup      = w_bias * w_sup_scale
    // ═══════════════════════════════════════════════════════════════

    // ── INT32 → FP32 per channel ──
    wire [`FP32_WIDTH-1:0] int_fp [0:N_CHAN-1];
    genvar g;
    generate
        for (g = 0; g < N_CHAN; g = g + 1) begin : int2fp_gen
            assign int_fp[g] = int32_to_fp32(partial_sums_in[g]);
        end
    endgenerate

    // ── Pre-computation: combined_scale = w_scale * w_sup_scale ──
    wire [`FP32_WIDTH-1:0] pre_comb_scale;
    fp32_mul u_mul_comb_scale (
        .a     (w_scale),
        .b     (w_sup_scale),
        .result(pre_comb_scale)
    );

    // ── Pre-computation: w_bias_sup = w_bias * w_sup_scale ──
    wire [`FP32_WIDTH-1:0] pre_w_bias_sup;
    fp32_mul u_mul_w_bias_sup (
        .a     (w_bias),
        .b     (w_sup_scale),
        .result(pre_w_bias_sup)
    );

    // ═══════════════════════════════════════════════════════════════
    // Stage 1 → Stage 2 pipeline registers
    // ═══════════════════════════════════════════════════════════════

    reg [`FP32_WIDTH-1:0] s1_int_fp [0:N_CHAN-1];
    reg [`FP32_WIDTH-1:0] s1_comb_scale;
    reg [`FP32_WIDTH-1:0] s1_w_bias_sup;
    reg [`FP32_WIDTH-1:0] s1_w_sup_bias;
    reg                   s1_valid;

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < N_CHAN; j = j + 1) begin
                s1_int_fp[j] <= 32'd0;
            end
            s1_comb_scale <= 32'd0;
            s1_w_bias_sup <= 32'd0;
            s1_w_sup_bias <= 32'd0;
            s1_valid      <= 1'b0;
        end else begin
            for (j = 0; j < N_CHAN; j = j + 1) begin
                s1_int_fp[j] <= int_fp[j];
            end
            s1_comb_scale <= pre_comb_scale;
            s1_w_bias_sup <= pre_w_bias_sup;
            s1_w_sup_bias <= w_sup_bias;  // Register for use in Stage 2
            s1_valid      <= valid_in;
        end
    end

    // ═══════════════════════════════════════════════════════════════
    // Stage 2: combinatorial logic
    //   (a) combined_bias = s1_w_bias_sup + s1_w_sup_bias
    //   (b) Per channel: s1_int_fp * s1_comb_scale + combined_bias
    // ═══════════════════════════════════════════════════════════════

    // ── combined_bias = s1_w_bias_sup + s1_w_sup_bias ──
    wire [`FP32_WIDTH-1:0] combined_bias;
    fp32_add u_add_comb_bias (
        .a     (s1_w_bias_sup),
        .b     (s1_w_sup_bias),
        .result(combined_bias)
    );

    // ── Per-channel: scaled = s1_int_fp[g] * s1_comb_scale ──
    wire [`FP32_WIDTH-1:0] s2_scaled [0:N_CHAN-1];
    generate
        for (g = 0; g < N_CHAN; g = g + 1) begin : s2_mul_gen
            fp32_mul u_mul_chan (
                .a     (s1_int_fp[g]),
                .b     (s1_comb_scale),
                .result(s2_scaled[g])
            );
        end
    endgenerate

    // ── Per-channel: final = s2_scaled[g] + combined_bias ──
    wire [`FP32_WIDTH-1:0] s2_result [0:N_CHAN-1];
    generate
        for (g = 0; g < N_CHAN; g = g + 1) begin : s2_add_gen
            fp32_add u_add_chan (
                .a     (s2_scaled[g]),
                .b     (combined_bias),
                .result(s2_result[g])
            );
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Stage 2 output registers
    // ═══════════════════════════════════════════════════════════════

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < N_CHAN; k = k + 1) begin
                results_out[k] <= 32'd0;
            end
            valid_out <= 1'b0;
        end else begin
            for (k = 0; k < N_CHAN; k = k + 1) begin
                results_out[k] <= s2_result[k];
            end
            valid_out <= s1_valid;
        end
    end

endmodule
