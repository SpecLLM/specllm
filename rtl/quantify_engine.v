// quantify_engine.v — Dynamic Activation Quantization Engine (FP16 → Q8K)
//
// Three-stage pipeline hiding quantization latency behind primary compute:
//   Stage 1 (COLLECT): Stream in GROUP_SIZE activations per channel.
//       Convert FP16→abs-magnitude, track per-channel running max via
//       O(1)-per-cycle comparator (equivalent to O(log N) reduction tree
//       amortized over GROUP_SIZE cycles). Buffer all values for drain.
//   Stage 2 (SCALE):   For each channel, generate 8-bit scale factor from
//       max_abs using priority-encoder + barrel-shift + small LUT.
//       Replaces FP division with shift-based approximation.
//   Stage 3 (DRAIN):   Read buffered activations, quantize via shift+clamp
//       using per-channel scale factors. Output Q8 INT8 streams.
//
// Parameters:
//   GROUP_SIZE   — activations per quantization group (shared scale factor)
//   NUM_CHANNELS — parallel input channels (independent scale per channel)

`include "sparsenn_params.vh"

module quantify_engine #(
    parameter GROUP_SIZE   = 128,      // Activations per quantization group
    parameter NUM_CHANNELS = 64        // Parallel input channels
) (
    input  wire                                    clk,
    input  wire                                    rst_n,

    // ── Input: streaming FP16 activations ──
    input  wire [NUM_CHANNELS-1:0][`FP16_WIDTH-1:0] fp16_activations,
    input  wire                                    valid_in,
    input  wire                                    start_group,

    // ── Output: quantized INT8 + per-channel scale factors ──
    output reg  [NUM_CHANNELS-1:0][7:0]            q8_activations,
    output reg  [NUM_CHANNELS-1:0][7:0]            scale_factors,
    output reg                                      valid_out,
    output reg                                      ready_out,

    // ── External buffer SRAM interface (128 deep × NUM_CHANNELS*16 wide) ──
    // Moved to top-level for hierarchical synthesis compatibility
    output reg                                      buf_sram_wr_en,
    output reg  [6:0]                               buf_sram_wr_addr,
    output reg  [NUM_CHANNELS*`FP16_WIDTH-1:0]      buf_sram_wr_data,
    output reg  [6:0]                               buf_sram_rd_addr,
    input  wire [NUM_CHANNELS*`FP16_WIDTH-1:0]      buf_sram_rd_data
);

    // ═══════════════════════════════════════════════════════════════
    // State encoding
    // ═══════════════════════════════════════════════════════════════

    localparam S_IDLE    = 3'd0;
    localparam S_COLLECT = 3'd1;
    localparam S_SCALE   = 3'd2;
    localparam S_DRAIN   = 3'd3;
    localparam S_DONE    = 3'd4;

    // ═══════════════════════════════════════════════════════════════
    // Internal registers
    // ═══════════════════════════════════════════════════════════════

    reg  [2:0]                          state, next_state;

    // Per-channel running max (FP16 magnitude: sign cleared, [14:0] = {exp, man})
    reg  [NUM_CHANNELS-1:0][14:0]       max_mag;

    // Per-channel scale factor (8-bit encoded: {shift[4:0], lut_corr[2:0]})
    reg  [NUM_CHANNELS-1:0][7:0]        scale_reg;

    // Group-element counter (0 .. GROUP_SIZE-1)
    reg  [`FP32_WIDTH-1:0]             elem_cnt;        // wide enough for GROUP_SIZE

    // ── Buffer read/write pointers (external SRAM addresses) ──
    reg  [`FP32_WIDTH-1:0]             buf_wr_ptr;
    reg  [`FP32_WIDTH-1:0]             buf_rd_ptr;

    // Buffer SRAM is instantiated at top level (128 deep × NUM_CHANNELS*16 wide)
    // Collocate with other SRAM macros to simplify hierarchical synthesis
    reg  [NUM_CHANNELS-1:0][14:0]       max_mag_s2;     // latched max for scale stage (loaded at end of COLLECT)

    // ═══════════════════════════════════════════════════════════════
    // Stage 1: FP16 abs-magnitude extraction (combinational)
    // ═══════════════════════════════════════════════════════════════

    // Clear sign bit to get unsigned magnitude encoding.
    // IEEE 754 sign-magnitude: clearing sign gives monotonic encoding,
    // so unsigned integer compare works for magnitude comparison.
    wire [NUM_CHANNELS-1:0][14:0] input_mag;
    genvar g;
    generate
        for (g = 0; g < NUM_CHANNELS; g = g + 1) begin : mag_extract
            assign input_mag[g] = fp16_activations[g][14:0];
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Stage 1 (cont.): Running-max update per channel (combinational)
    // Each cycle during COLLECT: max_mag[c] = max(max_mag[c], input_mag[c])
    // This is the temporal equivalent of a logarithmic reduction tree.
    // For GROUP_SIZE=128, the amortized comparison work is O(1) per cycle
    // instead of O(log N) per value in a spatial tree.
    // ═══════════════════════════════════════════════════════════════

    wire [NUM_CHANNELS-1:0][14:0] max_mag_next;
    generate
        for (g = 0; g < NUM_CHANNELS; g = g + 1) begin : max_cmp
            assign max_mag_next[g] = (input_mag[g] > max_mag[g]) ? input_mag[g] : max_mag[g];
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Stage 2: Scale factor generation — priority encoder + shift + LUT
    //
    // For symmetric INT8 quantization:
    //   scale = max_abs / 127.0
    //   q8    = round(activation / scale)
    //
    // Avoid division: find log2(max_abs), then shift ≈ log2(max_abs) - 7.
    //   max_abs (FP16) = (1.M) × 2^(E-15)  for normalized  (E > 0)
    //   max_abs (FP16) = (0.M) × 2^(-14)    for subnormal   (E = 0)
    //
    //   log2(max_abs) = E - 15 + log2(1.M)  for normalized
    //   log2(max_abs) = -14 + log2(0.M)     for subnormal
    //
    // Scale factor = {shift_amt[4:0], lut_idx[2:0]}:
    //   shift_amt = floor(log2(max_abs)) - 7  (clamped to 5-bit range)
    //   lut_idx   = top 3 mantissa bits after normalization (fine correction)
    // ═══════════════════════════════════════════════════════════════

    // ── Per-channel scale factor computation ──
    // Wires for the scale-compute datapath (combinational)
    wire [NUM_CHANNELS-1:0][4:0]  chan_exp;        // FP16 exponent [14:10]
    wire [NUM_CHANNELS-1:0][9:0]  chan_man;        // FP16 mantissa [9:0]
    wire [NUM_CHANNELS-1:0]       chan_is_norm;     // normalized? (exp > 0)
    wire [NUM_CHANNELS-1:0][10:0] chan_man_ext;     // extended mantissa {hidden, man}
    wire [NUM_CHANNELS-1:0][3:0]  chan_lod;         // leading-1 position in man_ext
    wire [NUM_CHANNELS-1:0][7:0]  chan_eff_exp;     // effective exponent (signed)
    wire [NUM_CHANNELS-1:0][4:0]  chan_shift_amt;   // right-shift amount for quantize
    wire [NUM_CHANNELS-1:0][2:0]  chan_lut_idx;     // LUT index for mantissa correction

    generate
        for (g = 0; g < NUM_CHANNELS; g = g + 1) begin : scale_gen

            // Decompose max_mag_s2 (FP16 magnitude, 15 bits)
            assign chan_exp[g] = max_mag_s2[g][14:10];
            assign chan_man[g] = max_mag_s2[g][9:0];
            assign chan_is_norm[g] = |chan_exp[g];

            // Extended mantissa: prepend hidden bit
            assign chan_man_ext[g] = chan_is_norm[g] ? {1'b1, chan_man[g]}
                                                      : {1'b0, chan_man[g]};

            // ── Priority encoder: find position of leading 1 in 11-bit man_ext ──
            // Returns 0-10 for normalized (hidden bit is bit 10), 0-9 for subnormal
            // lod[3:0] = position of MSB (10 = bit[10], 0 = bit[0] or all zeros)
            wire [10:0] me = chan_man_ext[g];
            wire [3:0]  lod;

            // Priority-encode tree for 11-bit LOD
            if (1) begin : lod_11bit
                reg [3:0] lod_r;
                always @(*) begin
                    if      (me[10]) lod_r = 4'd10;
                    else if (me[9])  lod_r = 4'd9;
                    else if (me[8])  lod_r = 4'd8;
                    else if (me[7])  lod_r = 4'd7;
                    else if (me[6])  lod_r = 4'd6;
                    else if (me[5])  lod_r = 4'd5;
                    else if (me[4])  lod_r = 4'd4;
                    else if (me[3])  lod_r = 4'd3;
                    else if (me[2])  lod_r = 4'd2;
                    else if (me[1])  lod_r = 4'd1;
                    else if (me[0])  lod_r = 4'd0;
                    else             lod_r = 4'd0;  // all zeros → position 0
                end
                assign lod = lod_r;
            end

            // ── Effective exponent (signed, 8-bit) ──
            // normalized:   log2(value) = (E - 15) + (lod - 10)
            //               = E + lod - 25
            // subnormal:    log2(value) = -14 + (lod - 10)  [lod in 0..9]
            //               = lod - 24
            //   Note: for subnormal, man_ext = {0, man}, so lod is 0..9.
            //   The implicit exponent is -14, and mantissa is 0.M = M/1024.
            //   log2(0.M) = log2(M) - 10. If leading 1 is at position lod:
            //   log2(0.M) ≈ lod - 10 (approximating with leading-1 position)
            //   Actually: value = M/1024 * 2^(-14) = M * 2^(-24)
            //   If M has leading 1 at position lod (0..9), then M ≈ 2^lod
            //   So log2(value) ≈ lod - 24
            wire [7:0] eff_exp_norm;   // = {3'd0, chan_exp[g]} + lod - 8'd25
            wire [7:0] eff_exp_sub;    // = {4'd0, lod} - 8'd24
            assign eff_exp_norm = {3'd0, chan_exp[g]} + {4'd0, lod} - 8'd25;
            assign eff_exp_sub  = {4'd0, lod} - 8'd24;
            assign chan_eff_exp[g] = chan_is_norm[g] ? eff_exp_norm : eff_exp_sub;

            // ── Shift amount for quantization ──
            // shift = log2(value) - log2(127) ≈ log2(value) - 7
            //        ≈ eff_exp - 7
            // Shift amount is signed conceptually, but we clamp to 0..31 for right-shift.
            // If eff_exp < 7, the max is small (<127) so we don't shift (use LUT only).
            // If eff_exp >> 7, we right-shift by up to 31 positions.
            wire [7:0] shift_signed = chan_eff_exp[g] - 8'd7;
            assign chan_shift_amt[g] = shift_signed[7] ? 5'd0           // negative → no shift
                                         : (shift_signed > 8'd31) ? 5'd31  // clamp
                                         : shift_signed[4:0];

            // ── LUT index: top 3 bits of the fractional mantissa after normalization ──
            // After shifting man_ext left so the leading 1 is at bit 10,
            // the next 3 bits (bits[9:7] or equivalent) give fine correction.
            // For normalized (lod=10): use man[9:7] directly
            // For subnormal or non-ideal: shift man_ext left by (10-lod), take bits[9:7]
            // Simplified: use original mantissa top 3 bits [9:7]
            assign chan_lut_idx[g] = chan_man[g][9:7];

        end
    endgenerate

    // ── LUT for mantissa correction (8 entries × 3 bits → fine adjustment) ──
    // Correction factor ≈ (1 + lut_idx/8) / 2^(lod_fractional)
    // This is a simplified 3-bit mantissa correction stored in scale factor low bits.
    // Full LUT would be instantiated per channel; simplified here for area estimation.
    // LUT index = chan_lut_idx[g] = chan_man[g][9:7] (top 3 mantissa bits).
    // The dequant path uses: val_dequant ≈ q8 × 2^shift × (1 + lut_idx/8) / 2

    // ── Assemble 8-bit scale factor ──
    wire [NUM_CHANNELS-1:0][7:0] scale_computed;
    generate
        for (g = 0; g < NUM_CHANNELS; g = g + 1) begin : scale_pack
            // Format: {shift_amt[4:0], lut_correction[2:0]}
            assign scale_computed[g] = {chan_shift_amt[g], chan_lut_idx[g]};
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Stage 3: Quantization — shift-based division + clamp
    //
    // q8 = clip(round(fp16_activation / scale), -127, 127)
    //
    // Using shift + LUT instead of FP division:
    //   1. Extract sign bit and magnitude from FP16 activation
    //   2. Normalize mantissa (prepend hidden bit)
    //   3. Left-align magnitude via exponent-based shift
    //   4. Right-shift by scale_shift_amt (≈ divide by scale)
    //   5. Apply LUT-based fine correction
    //   6. Apply sign, clamp to [-127, 127]
    // ═══════════════════════════════════════════════════════════════

    // ── Combinational quantization datapath ──
    wire [NUM_CHANNELS-1:0][15:0] buf_activation;  // current activation from buffer
    wire [NUM_CHANNELS-1:0]       act_sign;         // sign bit
    wire [NUM_CHANNELS-1:0][4:0]  act_exp;          // exponent
    wire [NUM_CHANNELS-1:0][9:0]  act_man;          // mantissa
    wire [NUM_CHANNELS-1:0]       act_is_norm;      // normalized?
    wire [NUM_CHANNELS-1:0][10:0] act_man_ext;      // extended mantissa
    wire [NUM_CHANNELS-1:0][4:0]  act_drain_shift;  // shift amount from scale_reg
    wire [NUM_CHANNELS-1:0][7:0]  mag_clamped;      // clamped magnitude
    wire [NUM_CHANNELS-1:0][7:0]  q8_signed;        // signed result

    generate
        for (g = 0; g < NUM_CHANNELS; g = g + 1) begin : quant_datapath

            // Buffer read data from external SRAM (top-level instantiated)
            // buf_sram_rd_data is the current row; extract per-channel slice
            assign buf_activation[g] = buf_sram_rd_data[g*`FP16_WIDTH +: `FP16_WIDTH];

            // Extract FP16 components
            assign act_sign[g]  = buf_activation[g][15];
            assign act_exp[g]   = buf_activation[g][14:10];
            assign act_man[g]   = buf_activation[g][9:0];
            assign act_is_norm[g] = |act_exp[g];

            // Extended mantissa with hidden bit
            assign act_man_ext[g] = act_is_norm[g] ? {1'b1, act_man[g]}
                                                    : {1'b0, act_man[g]};

            // Shift amount from scale factor (upper 5 bits)
            // scale_reg was latched in S_SCALE and is stable by S_DRAIN
            assign act_drain_shift[g] = scale_reg[g][7:3];

            // ── Magnitude scaling via barrel shift ──
            // The activation magnitude is approximately act_man_ext × 2^(act_exp-15+lod).
            // To divide by scale: right-shift by act_drain_shift.
            //
            // Approach: convert FP16 activation to fixed-point magnitude,
            // then right-shift by the scale-derived amount to quantize.
            // This replaces FP division with a single barrel-shift per channel.

            // Simple LOD for activation mantissa
            reg [3:0] act_lod;
            always @(*) begin
                if      (act_man_ext[g][10]) act_lod = 4'd10;
                else if (act_man_ext[g][9])  act_lod = 4'd9;
                else if (act_man_ext[g][8])  act_lod = 4'd8;
                else if (act_man_ext[g][7])  act_lod = 4'd7;
                else if (act_man_ext[g][6])  act_lod = 4'd6;
                else if (act_man_ext[g][5])  act_lod = 4'd5;
                else if (act_man_ext[g][4])  act_lod = 4'd4;
                else if (act_man_ext[g][3])  act_lod = 4'd3;
                else if (act_man_ext[g][2])  act_lod = 4'd2;
                else if (act_man_ext[g][1])  act_lod = 4'd1;
                else                          act_lod = 4'd0;
            end

            // Effective left-shift from exponent + mantissa position
            // eff_left_shift = (act_exp - 15) + (act_lod - 10) for normalized
            //                = act_exp + act_lod - 25  (same as eff_exp formula)
            // This can be negative (for small values)
            wire [7:0] act_eff_exp_norm = {3'd0, act_exp[g]} + {4'd0, act_lod} - 8'd25;
            wire [7:0] act_eff_exp_sub  = {4'd0, act_lod} - 8'd24;
            wire [7:0] act_eff_exp_full = act_is_norm[g] ? act_eff_exp_norm : act_eff_exp_sub;

            // Left-shift mantissa by act_eff_exp_full (clamped)
            // First, create a shifted version of man_ext
            wire [31:0] man_ext_32 = {21'd0, act_man_ext[g]};

            // Shift-left by act_eff_exp_full if positive, else shift-right
            wire [31:0] man_aligned;
            assign man_aligned = act_eff_exp_full[7] ? (man_ext_32 >> (~act_eff_exp_full[6:0] + 1'b1))
                                   : (man_ext_32 << act_eff_exp_full[6:0]);

            // Validation guard: limit to avoid excessive shift
            wire [31:0] man_aligned_safe;
            assign man_aligned_safe = (act_eff_exp_full[7] && act_eff_exp_full[6:0] > 7'd31) ? 32'd0
                                    : (!act_eff_exp_full[7] && act_eff_exp_full[6:0] > 7'd20) ? {man_ext_32[10:0], 21'd0}
                                    : man_aligned;

            // Now right-shift by scale-derived shift amount
            wire [15:0] mag_shifted;
            assign mag_shifted = man_aligned_safe[15:0] >> act_drain_shift[g];

            // ── Clamp magnitude to [0, 127] ──
            assign mag_clamped[g] = (|mag_shifted[15:7]) ? 8'd127 : mag_shifted[7:0];

            // ── Apply sign (symmetric quantization) ──
            assign q8_signed[g] = act_sign[g] ? (~mag_clamped[g] + 1'b1) : mag_clamped[g];

        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // FSM: Combinational next-state logic
    // ═══════════════════════════════════════════════════════════════

    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start_group && valid_in)
                    next_state = S_COLLECT;
            end

            S_COLLECT: begin
                if (valid_in && (elem_cnt == GROUP_SIZE - 1))
                    next_state = S_SCALE;
            end

            S_SCALE: begin
                next_state = S_DRAIN;
            end

            S_DRAIN: begin
                if (buf_rd_ptr == GROUP_SIZE - 1)
                    next_state = S_DONE;
            end

            S_DONE: begin
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    // ═══════════════════════════════════════════════════════════════
    // FSM: Sequential state + datapath
    // ═══════════════════════════════════════════════════════════════

    integer ci;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            elem_cnt    <= 32'd0;
            buf_wr_ptr  <= 32'd0;
            buf_rd_ptr  <= 32'd0;
            max_mag_s2  <= '0;

            for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                max_mag[ci]    <= 15'd0;
                scale_reg[ci]  <= 8'd0;
                scale_factors[ci] <= 8'd0;
                q8_activations[ci] <= 8'd0;
            end
            buf_sram_wr_en   <= 1'b0;
            buf_sram_wr_addr <= 7'd0;
            buf_sram_wr_data <= {(NUM_CHANNELS * `FP16_WIDTH){1'b0}};
            buf_sram_rd_addr <= 7'd0;
        end else begin
            state <= next_state;

            // Default SRAM control deassertions (overridden per-state)
            buf_sram_wr_en   <= 1'b0;
            buf_sram_wr_addr <= 7'd0;
            buf_sram_wr_data <= {(NUM_CHANNELS * `FP16_WIDTH){1'b0}};

            case (state)
                S_IDLE: begin
                    elem_cnt   <= 32'd0;
                    buf_wr_ptr <= 32'd0;
                    buf_rd_ptr <= 32'd0;
                    valid_out  <= 1'b0;
                    ready_out  <= 1'b1;

                    for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                        max_mag[ci] <= 15'd0;
                    end
                end

                S_COLLECT: begin
                    ready_out <= 1'b0;
                    valid_out <= 1'b0;

                    if (valid_in) begin
                        // Write incoming activations to external buffer SRAM
                        buf_sram_wr_en   <= 1'b1;
                        buf_sram_wr_addr <= buf_wr_ptr[6:0];
                        buf_sram_wr_data <= fp16_activations;

                        // Update per-channel running max (magnitude compare)
                        for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                            max_mag[ci] <= max_mag_next[ci];
                        end

                        // On last element of the group, latch final max_mag
                        // into max_mag_s2 for the scale-computation stage.
                        // max_mag_next includes the current cycle's input,
                        // so this captures the true group-wide maximum.
                        if (elem_cnt == GROUP_SIZE - 1) begin
                            for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                                max_mag_s2[ci] <= max_mag_next[ci];
                            end
                        end

                        buf_wr_ptr <= buf_wr_ptr + 32'd1;
                        elem_cnt   <= elem_cnt + 32'd1;
                    end
                end

                S_SCALE: begin
                    // max_mag_s2 was latched at end of S_COLLECT (previous cycle).
                    // scale_computed is purely combinational, driven by max_mag_s2.
                    // Latch the computed scale factors for output and quantization.
                    for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                        scale_reg[ci]    <= scale_computed[ci];
                        scale_factors[ci] <= scale_computed[ci];
                    end

                    // Reset pointers for drain phase
                    buf_wr_ptr <= 32'd0;
                    buf_rd_ptr <= 32'd0;
                    elem_cnt   <= 32'd0;
                end

                S_DRAIN: begin
                    ready_out <= 1'b0;

                    // Drive external SRAM read address
                    buf_sram_wr_en   <= 1'b0;
                    buf_sram_rd_addr <= buf_rd_ptr[6:0];

                    // scale_reg was latched in S_SCALE and is stable.
                    // q8_signed is combinational, driven by buf_sram_rd_data
                    // and scale_reg. Quantize and output current buffer entry.
                    for (ci = 0; ci < NUM_CHANNELS; ci = ci + 1) begin
                        q8_activations[ci] <= q8_signed[ci];
                    end

                    valid_out  <= 1'b1;
                    buf_rd_ptr <= buf_rd_ptr + 32'd1;
                end

                S_DONE: begin
                    valid_out <= 1'b0;
                    ready_out <= 1'b0;
                    buf_rd_ptr <= 32'd0;
                end

                default: ;
            endcase
        end
    end

    // ═══════════════════════════════════════════════════════════════
    // Output assertions for synthesis
    // ═══════════════════════════════════════════════════════════════

    // synopsys translate_off
    // Assertions for simulation only
    always @(posedge clk) begin
        if (state == S_DRAIN && !valid_out)
            $display("[QUANTIFY_ENGINE] Warning: drain state with valid_out low");
    end
    // synopsys translate_on

endmodule
