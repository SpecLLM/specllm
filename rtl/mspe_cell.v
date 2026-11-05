// mspe_cell.v — DynLogic Shift-Mux-Add PE, ★ 8-bit 权重输入 / 多 lane
//
// ── 与旧版的区别 (重要) ────────────────────────────────────────────────
//   旧版: `input wire [2:0] weight` —— **只有 1 条 lane**, 是设计文档口径的
//         1/4 (Q2K) 或 1/2 (Q3K)。文档
//         (`docs/superpowers/specs/2026-07-07-...-design.md:664`) 写的是
//             "MsPE Array (64×64) | DynLogic multiplier-free PE array.
//              Q2K→4-lane, Q3K→2-lane MAC streams"
//         新版按这个口径实现。
//
// ── 权重字的解释 (8 bit, 内部按 lane 分组) ─────────────────────────────
//   Q2K (mode=0): 4 × 2 bit → **4 条 lane**, 每条幅值 mag∈{0,1,2,3}
//                 ⟹ 贡献 {0, a, 2a, 4a}, 4 条相加
//                 可表示的权重 = 4 个 {0,1,2,4} 之和 ∈ {0,…,16}·a
//                 ★ 旧版是 {0,2a,4a,8a} —— 2 bit 里塞了 8a (2^3), 且丢掉 2^0,
//                   导致可表示权重只有偶数 {0,2,…,32}, 累加器 LSB 恒 0。
//   Q3K (mode=1): 2 × 3 bit → **2 条 lane**, 每 3 bit = {mag[1:0], sign}
//                 ⟹ 贡献 ±{0, a, 2a, 4a}, 两条相加 (共 8 档 = 3 bit ✓)
//                 (bit[7] 在 Q3K 下未使用)
//
//   乘法器仍然**不存在**: 每条 lane 是一个「移位选择器」(act<<mag),
//   lane 相和 = 权重 × 激活 —— 即把权重量化成「若干 2 的幂之和」。
//   8 bit 正好够编码 4 条 (Q2K) 或 2 条 (Q3K) 这样的项。
//
// ── 数据流 (与旧版一致) ────────────────────────────────────────────────
//   weight-stationary: `weight_load` 时锁存 8-bit 权重, 之后保持不变
//   激活每拍向右传, 部分和每拍向下累加

module mspe_cell (
    input  wire         clk,
    input  wire         rst_n,

    // ── Control ──
    input  wire         mode,            // 0 = Q2K (4×2b), 1 = Q3K (2×3b)
    input  wire         weight_load,     // 锁存 8-bit 权重 (weight-stationary)
    input  wire         accumulate,      // 1 = 累加, 0 = 重新开始

    // ── Data ──
    input  wire [7:0]   activation_in,
    input  wire [7:0]   weight,          // ★ 8 bit / PE (旧版是 [2:0])
    input  wire [31:0]  partial_sum_in,

    output reg  [7:0]   activation_out,
    output reg  [31:0]  partial_sum_out
);

    reg  [7:0]  weight_reg;       // 8-bit 驻留权重
    reg  [7:0]  activation_reg;
    reg  [31:0] accum_reg;

    // ── 移位候选 (★ 2026-10-08 修正候选集) ──
    //   旧: {0, 2a, 4a, 8a} —— 用 2 bit 编码了**指数 1..3**, 既丢掉 2^0 = a,
    //       又让 8a (= 2^3, 3 bit 的量) 出现在"2-bit 权重"里, 与 Q2K 的位宽不符。
    //   新: {0, a, 2a, 4a}  —— 2 bit 该有的样子 (0 + 2^0..2^2)。
    //   ⟹ 奇/小权重 (1, 2, 3 …) 现在表示得出来, 动态范围相应从 0..8 收到 0..4。
    wire [10:0] act_ext = {3'd0, activation_reg};
    wire [10:0] sh0 = 11'd0;             // 0      (mag=0)
    wire [10:0] sh1 = act_ext;           // a      (mag=1)   ★ 旧: act_ext << 1
    wire [10:0] sh2 = act_ext << 1;      // 2a     (mag=2)   ★ 旧: act_ext << 2
    wire [10:0] sh3 = act_ext << 2;      // 4a     (mag=3)   ★ 旧: act_ext << 3

    // ── 一条 lane: 2-bit 幅值选择 + 可选符号 ──
    function signed [12:0] lane_mac;
        input [1:0] mag;
        input       sgn;
        reg   [10:0] m;
        begin
            case (mag)
                2'd0:    m = sh0;
                2'd1:    m = sh1;
                2'd2:    m = sh2;
                default: m = sh3;
            endcase
            lane_mac = sgn ? -{2'b00, m} : {2'b00, m};
        end
    endfunction

    // ── Q2K: 4 条 lane, 每 2 bit 一个无符号幅值 ──
    wire signed [12:0] q2k0 = lane_mac(weight_reg[1:0], 1'b0);
    wire signed [12:0] q2k1 = lane_mac(weight_reg[3:2], 1'b0);
    wire signed [12:0] q2k2 = lane_mac(weight_reg[5:4], 1'b0);
    wire signed [12:0] q2k3 = lane_mac(weight_reg[7:6], 1'b0);

    // ── Q3K: 2 条 lane, 每 3 bit = {mag[1:0], sign} ──
    wire signed [12:0] q3k0 = lane_mac(weight_reg[1:0], weight_reg[2]);
    wire signed [12:0] q3k1 = lane_mac(weight_reg[4:3], weight_reg[5]);

    // ── lane 求和 (= 「权重 × 激活」的无乘法器近似) ──
    wire signed [14:0] prod_sum = mode
        ? (q3k0 + q3k1)                                  // 2 lane
        : (q2k0 + q2k1 + q2k2 + q2k3);                   // 4 lane
    wire signed [31:0] product_32 = {{17{prod_sum[14]}}, prod_sum};

    // ── Systolic 部分和 + 时间累加 ──
    wire signed [31:0] psum_next  = $signed(partial_sum_in) + product_32;
    wire signed [31:0] accum_next = accumulate ? ($signed(accum_reg) + product_32)
                                               : product_32;

    // ── Pipeline registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 8'd0; activation_reg <= 8'd0; accum_reg <= 32'd0;
        end else begin
            if (weight_load) weight_reg <= weight;       // weight-stationary: 只锁一次
            activation_reg <= activation_in;
            accum_reg      <= accum_next;
        end
    end

    // ── Output registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out <= 8'd0; partial_sum_out <= 32'd0;
        end else begin
            activation_out  <= activation_reg;
            partial_sum_out <= psum_next;
        end
    end

endmodule
