// dsa_controller.v — Sequence FSM orchestrating all DSA sub-modules
//
// Coordinates DMA data movement (weights, activations, results) through the
// TileLink slave interface, then sequences through the compute pipeline:
// Quantify → MsPS Array → QuanRedPipe Fusion → FP16 Array → Vector Unit.
// Provides start/done handshakes to each sub-module and drives SRAM control
// signals during compute phases.

`include "sparsenn_params.vh"

module dsa_controller #(
    parameter DATA_WIDTH = 64,
    parameter N_PE_ROWS  = 16,
    parameter M_PE_COLS  = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── CSR inputs (from tl_slave_if register file) ──
    input  wire                     csr_start,
    input  wire                     csr_reset,
    input  wire [1:0]               csr_mode,
    input  wire [31:0]              csr_weight_addr,
    input  wire [31:0]              csr_act_addr,
    input  wire [31:0]              csr_result_addr,
    input  wire [15:0]              csr_num_rows,
    input  wire [15:0]              csr_num_cols,
    input  wire [15:0]              csr_block_size,
    input  wire [31:0]              csr_w_scale,
    input  wire [31:0]              csr_w_bias,
    input  wire [31:0]              csr_w_sup_scale,
    input  wire [31:0]              csr_w_sup_bias,

    // ── Status outputs (to tl_slave_if STATUS register) ──
    output reg                      status_busy,
    output reg                      status_done,
    output reg                      status_error,

    // ── DMA control (to tl_slave_if DMA engine) ──
    output reg                      dma_start,
    output reg                      dma_rw,
    output reg  [1:0]               dma_target,
    output reg  [15:0]              dma_total_words,
    input  wire                     dma_done,
    input  wire                     dma_busy,

    // ── Sub-module control ──
    output reg                      mspe_start,
    output reg  [1:0]               mspe_mode,
    input  wire                     mspe_done,

    // ── MsPE 权重加载 (★ 新增) ──
    // MsPE 是 **weight-stationary** 的: 权重只在每次 `mspe_start` 时锁进 PE 寄存器,
    // 之后整个计算相位都不再读。所以 8192 bit 的权重 tile 不需要"每拍 8192 bit"的
    // SRAM 带宽 —— 用**寄存器缓冲**分 8 拍从 wt SRAM 宽读灌满即可。
    //   一拍宽读 = 8 scratchpad × 128 b = 1024 b  ⟹ 8 拍灌满 8192 b
    //   子模块加载 FSM 在 load_req 之后保持 wt_loading 8 拍, 再花 1 拍清标志
    //   ⟹ 控制器在本状态等 **9 拍**, 然后才进 S_COMPUTE_MSPS 拉 mspe_start。
    output reg                      mspe_wt_load_req,   // 1 拍脉冲, 启动宽读
    output reg  [15:0]              mspe_wt_base_addr,  // 本 tile 在 wt SRAM 的起始行


    output reg                      quan_red_pipe_start,
    input  wire                     quan_red_pipe_done,

    output reg                      quantify_engine_start,
    input  wire                     quantify_engine_done,

    output reg                      vector_unit_start,
    output reg  [2:0]               vector_unit_op,
    input  wire                     vector_unit_done,

    output reg                      fp16_array_start,
    input  wire                     fp16_array_done,

    // ── SRAM control (driven during compute phases) ──
    output reg                      sram_weight_rd_en,
    output reg  [15:0]              sram_weight_rd_addr,
    output reg                      sram_act_rd_en,
    output reg  [15:0]              sram_act_rd_addr,
    output reg                      sram_acc_wr_en,
    output reg  [15:0]              sram_acc_wr_addr,
    output reg                      sram_acc_rd_en,
    output reg  [15:0]              sram_acc_rd_addr,

    // ── Debug state ──
    output wire [3:0]               dbg_state
);

    // ── State encoding (one-hot safe, binary compact) ──
    localparam S_IDLE            = 4'd0;
    localparam S_LOAD_WEIGHTS    = 4'd1;
    localparam S_LOAD_ACTIVATIONS = 4'd2;
    localparam S_QUANTIFY        = 4'd3;
    localparam S_COMPUTE_MSPS    = 4'd4;
    localparam S_FUSION          = 4'd6;
    localparam S_FP16_COMPUTE   = 4'd10;
    localparam S_VECTOR          = 4'd7;
    localparam S_STORE_RESULTS   = 4'd8;
    localparam S_DONE            = 4'd9;
    localparam S_MSPS_WLOAD      = 4'd11;   // ★ 新增: MsPE 权重 8 拍宽读 → 寄存器缓冲

    // ── MsPE 权重加载计数器 ──
    //   子模块 FSM: load_req 之后 wt_loading 保持 8 拍 (cnt 0..7), 第 8 拍清标志
    //   ⟹ 控制器从拉 load_req 到确认加载完, 一共等 9 拍
    localparam WT_LOAD_CYCLES = 5'd9;
    reg  [4:0] wt_load_cnt;
    wire wt_load_cnt_done = (wt_load_cnt >= WT_LOAD_CYCLES);
    // 一个权重 tile = 32×32×8b = 8192 b = 8 行 × 8 scratchpad
    // ⚠️ K 方向的循环尚未实现 (一次调用只算一个 tile), 所以 base 恒 0;
    //    将来加 K 循环时, 每完成一个 tile 就 +8。
    localparam WT_TILE_ROWS = 16'd8;

    // ── DMA target encoding ──
    localparam DMA_TARG_WEIGHT = 2'd0;
    localparam DMA_TARG_ACT    = 2'd1;
    localparam DMA_TARG_ACC    = 2'd2;

    reg [3:0] state, next_state;
    assign dbg_state = state;

    // ── DMA word count computation ──
    // Use num_rows * num_cols as a reasonable transfer size.
    // 16-bit * 16-bit = 32-bit; truncate to 16-bit for counter.
    wire [15:0] mat_words = csr_num_rows * csr_num_cols;  // truncated product

    // ── Phase-entry detection ──
    // Tracks whether we have already asserted the start signal for the
    // current state. Cleared automatically on any state transition so
    // each new state gets its one-cycle start pulse.
    reg phase_started;
    reg [3:0] prev_state;

    // ── State machine sequential ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // ── Next-state logic ──
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:
                if (csr_start)
                    next_state = S_LOAD_WEIGHTS;

            S_LOAD_WEIGHTS:
                if (dma_done)
                    next_state = S_LOAD_ACTIVATIONS;

            S_LOAD_ACTIVATIONS:
                if (dma_done)
                    next_state = S_QUANTIFY;

            S_QUANTIFY:
                if (quantify_engine_done)
                    next_state = S_MSPS_WLOAD;      // ★ 先去加载权重, 再算

            S_MSPS_WLOAD:
                if (wt_load_cnt_done)
                    next_state = S_COMPUTE_MSPS;

            S_COMPUTE_MSPS:
                if (mspe_done)
                    next_state = S_FUSION;


            S_FUSION:
                if (quan_red_pipe_done)
                    next_state = S_FP16_COMPUTE;

            S_FP16_COMPUTE:
                if (fp16_array_done)
                    next_state = S_VECTOR;

            S_VECTOR:
                if (vector_unit_done)
                    next_state = S_STORE_RESULTS;

            S_STORE_RESULTS:
                if (dma_done)
                    next_state = S_DONE;

            S_DONE:
                // Return to IDLE after done is acknowledged
                if (!csr_start)
                    next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

    // ── Main execution logic (outputs + control) ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ── Status ──
            status_busy  <= 1'b0;
            status_done  <= 1'b0;
            status_error <= 1'b0;

            // ── DMA ──
            dma_start       <= 1'b0;
            dma_rw          <= 1'b0;
            dma_target      <= 2'd0;
            dma_total_words <= 16'd0;
            phase_started   <= 1'b0;
            prev_state      <= S_IDLE;

            // ── Sub-module starts ──
            mspe_start            <= 1'b0;
            mspe_mode             <= 2'd0;
            mspe_wt_load_req      <= 1'b0;      // ★
            mspe_wt_base_addr     <= 16'd0;     // ★
            wt_load_cnt           <= 5'd0;      // ★
            quan_red_pipe_start   <= 1'b0;
            quantify_engine_start <= 1'b0;
            vector_unit_start     <= 1'b0;
            vector_unit_op        <= 3'd0;
            fp16_array_start      <= 1'b0;

            // ── SRAM control ──
            sram_weight_rd_en   <= 1'b0;
            sram_weight_rd_addr <= 16'd0;
            sram_act_rd_en      <= 1'b0;
            sram_act_rd_addr    <= 16'd0;
            sram_acc_wr_en      <= 1'b0;
            sram_acc_wr_addr    <= 16'd0;
            sram_acc_rd_en      <= 1'b0;
            sram_acc_rd_addr    <= 16'd0;

        end else begin
            // ── Default deassertions (pulsed signals) ──
            dma_start            <= 1'b0;
            mspe_start           <= 1'b0;
            mspe_wt_load_req     <= 1'b0;       // ★ 1 拍脉冲
            quan_red_pipe_start  <= 1'b0;
            quantify_engine_start <= 1'b0;
            vector_unit_start    <= 1'b0;
            fp16_array_start     <= 1'b0;

            // ── Phase-entry detection: clear flag on state transition ──
            // phase_started is cleared whenever the state changes, so the
            // new state always sees !phase_started on its first cycle and
            // can issue its one-cycle start pulse.
            if (state != prev_state)
                phase_started <= 1'b0;
            prev_state <= state;

            case (state)
                // ──────────────────────────────────────────────
                // IDLE: wait for start from CSR
                // ──────────────────────────────────────────────
                S_IDLE: begin
                    status_busy  <= 1'b0;
                    status_done  <= 1'b0;
                    status_error <= 1'b0;

                    sram_weight_rd_en   <= 1'b0;
                    sram_weight_rd_addr <= 16'd0;
                    sram_act_rd_en      <= 1'b0;
                    sram_act_rd_addr    <= 16'd0;
                    sram_acc_wr_en      <= 1'b0;
                    sram_acc_wr_addr    <= 16'd0;
                    sram_acc_rd_en      <= 1'b0;
                    sram_acc_rd_addr    <= 16'd0;

                    if (csr_start) begin
                        status_busy <= 1'b1;
                    end
                end

                // ──────────────────────────────────────────────
                // LOAD_WEIGHTS: DMA weights from DRAM → Weight SRAM
                // ──────────────────────────────────────────────
                S_LOAD_WEIGHTS: begin
                    if (!phase_started) begin
                        dma_start       <= 1'b1;
                        dma_rw          <= 1'b0;          // LOAD: host write → SRAM
                        dma_target      <= DMA_TARG_WEIGHT;
                        dma_total_words <= mat_words;
                        phase_started   <= 1'b1;
                    end
                end

                // ──────────────────────────────────────────────
                // LOAD_ACTIVATIONS: DMA activations → Activation SRAM
                // ──────────────────────────────────────────────
                S_LOAD_ACTIVATIONS: begin
                    if (!phase_started) begin
                        dma_start       <= 1'b1;
                        dma_rw          <= 1'b0;          // LOAD
                        dma_target      <= DMA_TARG_ACT;
                        dma_total_words <= mat_words;
                        phase_started   <= 1'b1;
                    end
                end

                // ──────────────────────────────────────────────
                // QUANTIFY: Quantify Engine on activations
                // ──────────────────────────────────────────────
                S_QUANTIFY: begin
                    if (!phase_started) begin
                        quantify_engine_start <= 1'b1;
                        phase_started <= 1'b1;
                    end
                    // Quantify Engine accesses activation SRAM internally;
                    // provide basic read-enable for SRAM window
                    sram_act_rd_en   <= 1'b1;
                    sram_act_rd_addr <= 16'd0;  // base address
                end

                // ──────────────────────────────────────────────
                // MSPS_WLOAD: MsPE 权重 8 拍宽读 → 8192-bit 寄存器缓冲 ★ 新增
                // ──────────────────────────────────────────────
                //   weight-stationary ⟹ 权重不需要每拍 8192 bit 的 SRAM 带宽,
                //   分 8 拍灌进缓冲即可 (稳态端口占用 8/64 = 12.5%)。
                //   本状态: 第 1 拍拉 load_req, 之后等 9 拍让子模块加载完。
                S_MSPS_WLOAD: begin
                    if (!phase_started) begin
                        mspe_wt_load_req <= 1'b1;       // 1 拍脉冲, 启动宽读
                        mspe_wt_base_addr <= 16'd0;     // K 循环未实现, base 恒 0
                        wt_load_cnt      <= 5'd0;
                        phase_started    <= 1'b1;
                    end else begin
                        mspe_wt_load_req <= 1'b0;       // 脉冲只持续一拍
                        if (!wt_load_cnt_done)
                            wt_load_cnt <= wt_load_cnt + 5'd1;
                    end
                    // 加载期间让 wt SRAM 处于宽读状态 (csb 会把 8 个一起使能)
                    sram_weight_rd_en   <= 1'b1;
                end

                // ──────────────────────────────────────────────
                // COMPUTE_MSPS: MsPE array compute (Q2K / Q3K)
                // ──────────────────────────────────────────────
                S_COMPUTE_MSPS: begin
                    if (!phase_started) begin
                        mspe_start  <= 1'b1;
                        mspe_mode   <= csr_mode;
                        phase_started <= 1'b1;
                    end

                    // MsPE reads weights and activations from SRAMs
                    sram_weight_rd_en   <= 1'b1;
                    sram_weight_rd_addr <= 16'd0;
                    sram_act_rd_en      <= 1'b1;
                    sram_act_rd_addr    <= 16'd0;
                end

                // ──────────────────────────────────────────────
                // FUSION: QuanRedPipe dequant fusion
                // ──────────────────────────────────────────────
                S_FUSION: begin
                    if (!phase_started) begin
                        quan_red_pipe_start <= 1'b1;
                        phase_started       <= 1'b1;
                    end

                    // QuanRedPipe reads and writes accumulator SRAM
                    sram_acc_rd_en   <= 1'b1;
                    sram_acc_rd_addr <= 16'd0;
                    sram_acc_wr_en   <= 1'b1;
                    sram_acc_wr_addr <= 16'd0;
                end

                // ──────────────────────────────────────────────
                // FP16_COMPUTE: FP16 array dense matmul
                // ──────────────────────────────────────────────
                S_FP16_COMPUTE: begin
                    if (!phase_started) begin
                        fp16_array_start <= 1'b1;
                        phase_started   <= 1'b1;
                    end

                    // FP16 array reads weights and activations from SRAMs
                    sram_weight_rd_en   <= 1'b1;
                    sram_weight_rd_addr <= 16'd0;
                    sram_act_rd_en      <= 1'b1;
                    sram_act_rd_addr    <= 16'd0;
                end

                // ──────────────────────────────────────────────
                // VECTOR: Vector Unit element-wise post-processing
                // ──────────────────────────────────────────────
                S_VECTOR: begin
                    if (!phase_started) begin
                        vector_unit_start <= 1'b1;
                        vector_unit_op    <= 3'd0;     // default: element-wise op
                        phase_started     <= 1'b1;
                    end

                    // Vector Unit reads and writes accumulator SRAM
                    sram_acc_rd_en   <= 1'b1;
                    sram_acc_rd_addr <= 16'd0;
                    sram_acc_wr_en   <= 1'b1;
                    sram_acc_wr_addr <= 16'd0;
                end

                // ──────────────────────────────────────────────
                // STORE_RESULTS: DMA results from Acc SRAM → DRAM
                // ──────────────────────────────────────────────
                S_STORE_RESULTS: begin
                    if (!phase_started) begin
                        dma_start       <= 1'b1;
                        dma_rw          <= 1'b1;        // STORE: host read ← SRAM
                        dma_target      <= DMA_TARG_ACC;
                        dma_total_words <= mat_words;
                        phase_started   <= 1'b1;
                    end
                end

                // ──────────────────────────────────────────────
                // DONE: assert done flag, clear on next IDLE
                // ──────────────────────────────────────────────
                S_DONE: begin
                    status_busy <= 1'b0;
                    status_done <= 1'b1;

                    // Deassert all SRAM controls
                    sram_weight_rd_en   <= 1'b0;
                    sram_act_rd_en      <= 1'b0;
                    sram_acc_wr_en      <= 1'b0;
                    sram_acc_rd_en      <= 1'b0;
                end

                default: ;
            endcase
        end
    end

endmodule
