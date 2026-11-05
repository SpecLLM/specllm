// mspe_array_32x32.v — 32×32 MsPE Systolic Array (DynLogic, multiplier-free)
//
// 几何 **32×32** (PE 数 1024, 列数 32)。每 PE 权重字 **8 bit**, 内部按 lane
// 分组解释 (Q2K: 4×2b / Q3K: 2×3b) —— 见 mspe_cell.v。
//
// ── ★ 权重端口 1024 bit ────────────────────────────────────────────────
//   MsPE 是 **weight-stationary** 的: 权重只在 `start` 时锁进 PE, 之后 64 拍计算
//   全程不再读。所以 32×32×8b = 8192 bit 根本不需要"一拍并行进来" ——
//   **用寄存器缓冲分 8 拍从 SRAM 宽读灌进去就行**。
//
//   若把 32×32×8b = 8192 位做成**模块端口**, 后果有两个:
//     ① 层次化时这个 block 会有 **9,477 个 IO 引脚** (5 + 256 + 8192 + 1024),
//        而 1300² die 的周长只放得下 9,272 个 ⟹ `PPL-0024` 直接失败;
//     ② DSA 顶层还要把 8192 根线**穿过宏边界**布进去, 布线代价极高。
//   收窄到 1024 bit 后: 引脚数 9,477 → **2,314**, 且跨宏线减少 **87.5%**。
//
// ── 权重加载时序 (10 拍) ────────────────────────────────────────────────
//   sratchpad SRAM 是**一拍延迟**读 (dout 寄存器输出), 所以地址要提前一拍发。
//   数组把 `wt_load_idx` 组合输出给上层拼 SRAM 行地址, 内部用 `idx_d` (打一拍)
//   决定这拍回来的 `wt_wide` 写进哪个字:
//
//     拍 0   req=1           → busy<=1, idx<=0           (上层开始驱动 addr=idx)
//     拍 1   addr=0 已发      → idx_d=0, 写 word 0
//     拍 2   addr=1          → idx_d=0, 写 word 0 (addr0 的数据)  ... 依次类推
//     ...
//     拍 9   最后一个字       → busy<=0
//
//   ⟹ 上层控制器等 **10 拍** (WT_LOAD_CYCLES = 10)。
//   稳态占用: 10/64 = **15.6%**; 想完全藏掉可以再叠一层双缓冲 (本版没做)。

module mspe_array_32x32 #(
    parameter integer N_ROWS = 32,        // 行数 (= 激活向量宽度)
    parameter integer M_COLS = 32          // 列数 (= 部分和个数)
) (
    input  wire                                 clk,
    input  wire                                 rst_n,

    // ── Control ──
    input  wire                                 mode,        // 0 = Q2K (4 lane), 1 = Q3K (2 lane)
    input  wire                                 start,       // 拍一拍: 把 wt_buf 锁进所有 PE
    input  wire                                 accumulate,  // 1 = 跨拍累加, 0 = 重新开始

    // ── ★ 权重加载 (8 拍宽读, 端口只有 1024 bit) ──
    input  wire                                 wt_load_req,  // 1 拍脉冲, 启动加载
    input  wire [1023:0]                        wt_wide,      // 一拍宽读回来的 1024 bit
    output reg                                  wt_load_busy, // 加载中
    output wire [2:0]                           wt_load_idx,  // 0..7, 上层据此拼 SRAM 行地址

    // ── Activation input (左边缘, N_ROWS × 8b) ──
    input  wire [N_ROWS-1:0][7:0]               activations,

    // ── Partial sum output (下边缘, M_COLS × 32b) ──
    output wire [M_COLS-1:0][31:0]              partial_sums
);

    localparam integer NBIT = N_ROWS * M_COLS * 8;      // 权重缓冲总位数 (32×32×8 = 8192)

    // ══════════════════════════════════════════════════════════════
    // ★ 权重寄存器缓冲 + 8 拍加载 FSM
    // ══════════════════════════════════════════════════════════════
    //   一个权重 tile = N_ROWS × M_COLS × 8b = 8192 b
    //   一拍宽读 = 8 scratchpad × 128 b = 1024 b ⟹ 8 拍
    reg [NBIT-1:0] wt_buf;
    reg [2:0]      idx_r;      // 本拍驱动的地址下标 (组合输出)
    reg [2:0]      idx_d;      // idx_r 打一拍 = 本拍 wt_wide 对应的字

    assign wt_load_idx = idx_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_load_busy <= 1'b0;
            idx_r        <= 3'd0;
            idx_d        <= 3'd0;
            wt_buf       <= {NBIT{1'b0}};
        end else if (!wt_load_busy) begin
            if (wt_load_req) begin
                wt_load_busy <= 1'b1;
                idx_r        <= 3'd0;
            end
        end else begin
            // 本拍的 wt_wide 是**上一拍地址** idx_d 的数据
            wt_buf[idx_d*1024 +: 1024] <= wt_wide;
            idx_d <= idx_r;
            if (idx_r == 3'd7) begin
                wt_load_busy <= 1'b0;              // 8 个字收完
            end else begin
                idx_r <= idx_r + 3'd1;
            end
        end
    end

    // ══════════════════════════════════════════════════════════════
    // 权重分发: PE(i,j) 拿 tile 内第 (i*M_COLS + j) 个**字节**
    //   ★ 旧版是 `(wi*16+wj)%8` 那套 3-bit 跨字段切片 + 与行号无关的别名,
    //     8-bit 字节对齐之后两个问题同时消失。
    // ══════════════════════════════════════════════════════════════
    wire [N_ROWS-1:0][M_COLS-1:0][7:0] mspe_weights;
    genvar wi, wj;
    generate
        for (wi = 0; wi < N_ROWS; wi = wi + 1) begin : wt_row
            for (wj = 0; wj < M_COLS; wj = wj + 1) begin : wt_col
                assign mspe_weights[wi][wj] = wt_buf[(wi*M_COLS + wj)*8 +: 8];
            end
        end
    endgenerate

    // ══════════════════════════════════════════════════════════════
    // Systolic 网格 (与旧版一致)
    // ══════════════════════════════════════════════════════════════
    wire [7:0]  activation_wire  [0:N_ROWS-1][0:M_COLS-1];
    wire [31:0] partial_sum_wire [0:N_ROWS-1][0:M_COLS-1];

    generate
        for (wi = 0; wi < N_ROWS; wi = wi + 1) begin : row
            for (wj = 0; wj < M_COLS; wj = wj + 1) begin : col

                // ── 激活输入: 左边缘取阵列输入, 内部取左邻居 ──
                wire [7:0] act_in_wire;
                if (wj == 0)
                    assign act_in_wire = activations[wi];
                else
                    assign act_in_wire = activation_wire[wi][wj-1];

                // ── 部分和输入: 上边缘补 0 (新的一轮累加), 内部取上邻居 ──
                wire [31:0] psum_in_wire;
                if (wi == 0)
                    assign psum_in_wire = 32'd0;
                else
                    assign psum_in_wire = partial_sum_wire[wi-1][wj];

                // ── PE ──
                mspe_cell u_mspe (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .mode            (mode),
                    .weight_load     (start),
                    .accumulate      (accumulate),
                    .activation_in   (act_in_wire),
                    .weight          (mspe_weights[wi][wj]),
                    .partial_sum_in  (psum_in_wire),
                    .activation_out  (activation_wire[wi][wj]),
                    .partial_sum_out (partial_sum_wire[wi][wj])
                );

            end
        end

        // ── 输出: 每列下边缘 ──
        // ⚠️ 历史注记: 旧候选集 {0, 2a, 4a, 8a} 全是偶数 ⟹ 累加和 LSB 结构性
        //    为 0, hilomap 会挑 `partial_sums[j][0]` 当全局常量源, 网表里于是
        //    出现"tie 单元驱动输出端口" -> OpenROAD 建 GROUND 网 -> DRT-0305。
        //    ★ 2026-10-08 候选集改成 {0, a, 2a, 4a} 后 LSB 不再是常量, 该结构
        //    自然消失。后处理 `scripts/fix_tie_ports.py` 保留作兜底 (无匹配时
        //    打印"无需修改"并原样返回)。
        for (wj = 0; wj < M_COLS; wj = wj + 1) begin : output_col
            assign partial_sums[wj] = partial_sum_wire[N_ROWS-1][wj];
        end
    endgenerate

endmodule
