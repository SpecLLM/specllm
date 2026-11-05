// tl_slave_if.v — TileLink Uncached Lightweight slave interface for DSA
//
// Provides CSR configuration register file (0x0000-0x0FFF) and DMA streaming
// data window (0x1000-0x1FFF). Handles TL-UL channel A requests and generates
// channel D responses. Internal DMA engine manages address generation and
// SRAM data routing during LOAD/STORE phases.
//
// ── 2026-10-09: DMA 写改 posted（互连瓶颈修复 A）──────────────────────────
// 原实现：TL FSM 单笔未完成 —— 一笔 DMA 写要 accept→RESP→D 握手才收下一笔，
// 直接卡死 CPU→DSA 的灌数吞吐。现改为：DMA 页的 PutFullData 走**旁路流水通路**
// （不经过 TL FSM），每拍 accept 1 笔、SPM 每拍写 1 word；D 应答由一个小 FIFO
// 独立产生（AccessAck，与 A 解耦）。CSR 访问与 DMA 读（STORE）仍在原 FSM 上
// 单笔服务，语义不变。

`include "sparsenn_params.vh"

module tl_slave_if #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 64
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── TileLink-UL Channel A (request) ──
    input  wire                     tl_a_valid,
    output wire                     tl_a_ready,
    input  wire [2:0]               tl_a_opcode,
    input  wire [2:0]               tl_a_param,
    input  wire [3:0]               tl_a_size,
    input  wire [1:0]               tl_a_source,
    input  wire [ADDR_WIDTH-1:0]    tl_a_address,
    input  wire [7:0]               tl_a_mask,
    input  wire [DATA_WIDTH-1:0]    tl_a_data,

    // ── TileLink-UL Channel D (response) ──
    output wire                     tl_d_valid,
    output wire [2:0]               tl_d_opcode,
    output wire [1:0]               tl_d_param,
    output wire [3:0]               tl_d_size,
    output wire [1:0]               tl_d_source,
    output wire                     tl_d_sink,
    output wire [DATA_WIDTH-1:0]    tl_d_data,
    output wire                     tl_d_error,
    input  wire                     tl_d_ready,

    // ── CSR register outputs (to dsa_controller) ──
    output wire                     csr_start,
    output wire                     csr_reset,
    output wire [1:0]               csr_mode,
    output wire [31:0]              csr_weight_addr,
    output wire [31:0]              csr_act_addr,
    output wire [31:0]              csr_result_addr,
    output wire [15:0]              csr_num_rows,
    output wire [15:0]              csr_num_cols,
    output wire [15:0]              csr_block_size,
    output wire [31:0]              csr_w_scale,
    output wire [31:0]              csr_w_bias,
    output wire [31:0]              csr_w_sup_scale,
    output wire [31:0]              csr_w_sup_bias,

    // ── Status inputs (from dsa_controller) ──
    input  wire                     status_busy,
    input  wire                     status_done,
    input  wire                     status_error,

    // ── DMA control (from dsa_controller) ──
    input  wire                     dma_start,
    input  wire                     dma_rw,           // 0=write-to-SRAM, 1=read-from-SRAM
    input  wire [1:0]               dma_target,       // 00=weight, 01=act, 10=acc
    input  wire [15:0]              dma_total_words,
    output reg                      dma_done,
    output reg                      dma_busy,

    // ── SRAM DMA interfaces (driven during DMA operations) ──
    output reg                      dma_sram_wt_wr_en,
    output reg  [15:0]              dma_sram_wt_wr_addr,
    output reg  [DATA_WIDTH-1:0]    dma_sram_wt_wr_data,
    output reg                      dma_sram_act_wr_en,
    output reg  [15:0]              dma_sram_act_wr_addr,
    output reg  [DATA_WIDTH-1:0]    dma_sram_act_wr_data,
    output reg                      dma_sram_acc_rd_en,
    output reg  [15:0]              dma_sram_acc_rd_addr,
    input  wire [DATA_WIDTH-1:0]    dma_sram_acc_rd_data
);

    // ── TL-UL Opcodes ──
    localparam TL_OP_GET           = 3'd4;
    localparam TL_OP_PUTFULLDATA   = 3'd0;
    localparam TL_OP_ACCESSACK     = 3'd0;
    localparam TL_OP_ACCESSACKDATA = 3'd1;

    // ── CSR Address map (byte offsets) ──
    localparam CSR_CTRL          = 8'h00;
    localparam CSR_STATUS        = 8'h04;
    localparam CSR_WEIGHT_ADDR   = 8'h08;
    localparam CSR_ACT_ADDR      = 8'h0C;
    localparam CSR_RESULT_ADDR   = 8'h10;
    localparam CSR_NUM_ROWS      = 8'h14;
    localparam CSR_NUM_COLS      = 8'h18;
    localparam CSR_BLOCK_SIZE    = 8'h1C;
    localparam CSR_W_SCALE       = 8'h20;
    localparam CSR_W_BIAS        = 8'h24;
    localparam CSR_W_SUP_SCALE   = 8'h28;
    localparam CSR_W_SUP_BIAS    = 8'h2C;

    // ── 8B 对齐别名 (★ 2026-10-10): 4B 偏移在 64-bit 子总线上不可达 ——
    //    sbus(128b)→cbus(64b) 的宽度转换会把 addr[2:0] 折进 mask, 从口看到的地址
    //    被重定到 8B 边界, 于是 0x14 的 NUM_ROWS 永远解不到。下面给出 8B 对齐别名,
    //    让 CSR 在真实互连 (含宽度转换) 上可达。原偏移保留, 不删。
    localparam CSR_NUM_ROWS_A    = 8'h28;   // alias of 0x14
    localparam CSR_NUM_COLS_A    = 8'h30;   // alias of 0x18
    localparam CSR_BLOCK_SIZE_A  = 8'h38;   // alias of 0x1C

    // ── Address space boundaries ──
    localparam CSR_BASE  = 12'h000;
    localparam CSR_MASK  = 12'hFFF;
    localparam DMA_BASE  = 12'h100;
    localparam DMA_MASK  = 12'hFFF;   // DMA occupies 0x1000-0x1FFF

    // ── TL FSM states ──
    localparam TL_IDLE = 2'd0;
    localparam TL_WAIT = 2'd1;   // wait for SRAM read data (DMA read path)
    localparam TL_RESP = 2'd2;

    // ── DMA FSM states ──
    localparam DMA_IDLE    = 2'd0;
    localparam DMA_RUNNING = 2'd1;

    // ── CSR Register File ──
    reg [DATA_WIDTH-1:0] rf_ctrl;
    reg [DATA_WIDTH-1:0] rf_status;
    reg [DATA_WIDTH-1:0] rf_weight_addr;
    reg [DATA_WIDTH-1:0] rf_act_addr;
    reg [DATA_WIDTH-1:0] rf_result_addr;
    reg [DATA_WIDTH-1:0] rf_num_rows;
    reg [DATA_WIDTH-1:0] rf_num_cols;
    reg [DATA_WIDTH-1:0] rf_block_size;
    reg [DATA_WIDTH-1:0] rf_w_scale;
    reg [DATA_WIDTH-1:0] rf_w_bias;
    reg [DATA_WIDTH-1:0] rf_w_sup_scale;
    reg [DATA_WIDTH-1:0] rf_w_sup_bias;

    // ── TL FSM registers ──
    reg [1:0]  tl_state;
    reg [1:0]  tl_next_state;
    reg        req_is_read;
    reg        req_is_csr;
    reg        req_is_dma;
    reg [7:0]  req_csr_addr;
    reg [2:0]  req_opcode;
    reg [1:0]  req_param;
    reg [3:0]  req_size;
    reg [1:0]  req_source;
    reg [DATA_WIDTH-1:0] req_wdata;
    reg [7:0]  req_mask;

    // ── DMA FSM registers ──
    reg [1:0]  dma_state;
    reg [15:0] dma_word_cnt;
    reg        dma_done_pulse;

    // ── D 应答旁路 FIFO (posted DMA 写用) ──
    //   每项 = {param[1:0], size[3:0], source[1:0]}；AccessAck 无数据。
    localparam DFIFO_AW = 3;
    reg  [7:0]           dfifo_mem [0:(1<<DFIFO_AW)-1];
    reg  [DFIFO_AW:0]    dfifo_wp, dfifo_rp;
    wire [DFIFO_AW:0]    dfifo_cnt   = dfifo_wp - dfifo_rp;
    wire                 dfifo_empty = (dfifo_cnt == 0);
    wire                 dfifo_full  = (dfifo_cnt == (1<<DFIFO_AW));
    wire [7:0]           dfifo_rdata = dfifo_mem[dfifo_rp[DFIFO_AW-1:0]];

    // ── Address decode (combinational) ──
    wire [11:0] a_page = tl_a_address[11:0];
    wire        a_is_csr  = (a_page[11:8] == 4'h0);                // 0x000-0xFFF
    wire        a_is_dma  = (a_page[11:8] == 4'h1);                // 0x1000-0x1FFF
    wire        a_is_read = (tl_a_opcode == TL_OP_GET);
    wire        a_is_write = (tl_a_opcode == TL_OP_PUTFULLDATA);

    // ── 请求分类（posted 写 vs FSM 服务）──
    wire        fsm_idle   = (tl_state == TL_IDLE);
    //   DMA 页的 PutFullData（LOAD 模式）→ 旁路流水通路
    wire        dma_wr_req = tl_a_valid && a_is_write && a_is_dma &&
                             (dma_state == DMA_RUNNING) && (dma_rw == 1'b0);
    wire        dma_wr_acc = dma_wr_req && ~dfifo_full;          // A/D 解耦，FIFO 有空位即收
    //   FSM 服务：CSR、DMA 读（STORE），以及 DM A 写以外的全部
    wire        fsm_acc    = tl_a_valid && fsm_idle && ~dma_wr_req;
    //   STORE 模式的 DMA 读（走 FSM）
    wire        dma_rd_acc = fsm_acc && a_is_read && a_is_dma && (dma_rw == 1'b1)
                           && (dma_state == DMA_RUNNING);
    //   统一的"一笔 DMA 事务被受理"脉冲（LOAD 写 或 STORE 读）
    wire        dma_count  = dma_wr_acc || dma_rd_acc;

    // ── TL a_ready ──
    //   DMA 写：只要 FIFO 未满就收（与 FSM 是否忙无关）
    //   其它请求：FSM 空闲才收（保持单笔未完成语义）
    assign      tl_a_ready = dma_wr_req ? ~dfifo_full : fsm_idle;

    // ── CSR register read mux ──
    reg [DATA_WIDTH-1:0] csr_read_data;

    always @(*) begin
        case (req_csr_addr)
            CSR_CTRL:        csr_read_data = rf_ctrl;
            CSR_STATUS:      csr_read_data = rf_status;
            CSR_WEIGHT_ADDR: csr_read_data = rf_weight_addr;
            CSR_ACT_ADDR:    csr_read_data = rf_act_addr;
            CSR_RESULT_ADDR: csr_read_data = rf_result_addr;
            CSR_NUM_ROWS:    csr_read_data = rf_num_rows;
            CSR_NUM_ROWS_A:  csr_read_data = rf_num_rows;
            CSR_NUM_COLS:    csr_read_data = rf_num_cols;
            CSR_NUM_COLS_A:  csr_read_data = rf_num_cols;
            CSR_BLOCK_SIZE:  csr_read_data = rf_block_size;
            CSR_BLOCK_SIZE_A: csr_read_data = rf_block_size;
            CSR_W_SCALE:     csr_read_data = rf_w_scale;
            CSR_W_BIAS:      csr_read_data = rf_w_bias;
            CSR_W_SUP_SCALE: csr_read_data = rf_w_sup_scale;
            CSR_W_SUP_BIAS:  csr_read_data = rf_w_sup_bias;
            default:         csr_read_data = {DATA_WIDTH{1'b0}};
        endcase
    end

    // ── TL State machine sequential ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            tl_state <= TL_IDLE;
        else
            tl_state <= tl_next_state;
    end

    // ── TL Next-state logic ──
    always @(*) begin
        tl_next_state = tl_state;
        case (tl_state)
            TL_IDLE: begin
                if (fsm_acc) begin
                    // For DMA reads, need one cycle to fetch SRAM data
                    if (a_is_read && a_is_dma && dma_state == DMA_RUNNING && dma_rw)
                        tl_next_state = TL_WAIT;
                    else
                        tl_next_state = TL_RESP;
                end
            end
            TL_WAIT: begin
                // One cycle read latency from SRAM
                tl_next_state = TL_RESP;
            end
            TL_RESP: begin
                if (tl_d_ready)          // D 已在 RESP 期间组合断言
                    tl_next_state = TL_IDLE;
            end
            default: tl_next_state = TL_IDLE;
        endcase
    end

    // ── TL Request capture (FSM 路径) ──
    //   注意: 输出 tl_d_* 由 FSM 的 D(组合) 与旁路 FIFO 一起 mux。
    //   ★ 握手 bug 修复: 原版把 tl_d_valid 注册后在**同一拍**按 tl_d_ready 清零,
    //     若主机一直拉高 d_ready(合法 TL 行为) 则 D 永不断言 → 死锁。
    //     现改为: state==TL_RESP 期间**组合**断言 valid, 握手成功才离开 RESP。
    wire        fsm_d_valid  = (tl_state == TL_RESP);
    wire [2:0]  fsm_d_opcode = req_is_read ? TL_OP_ACCESSACKDATA : TL_OP_ACCESSACK;
    wire [1:0]  fsm_d_param  = req_param;
    wire [3:0]  fsm_d_size   = req_size;
    wire [1:0]  fsm_d_source = req_source;
    wire        fsm_d_sink   = 1'b0;
    wire        fsm_d_error  = 1'b0;
    wire [DATA_WIDTH-1:0] fsm_d_data = req_is_read
                                     ? (req_is_csr ? csr_read_data : dma_sram_acc_rd_data)
                                     : {DATA_WIDTH{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_opcode  <= 3'd0;
            req_param   <= 2'd0;
            req_size    <= 4'd0;
            req_source  <= 2'd0;
            req_wdata   <= {DATA_WIDTH{1'b0}};
            req_mask    <= 8'd0;
            req_is_read <= 1'b0;
            req_is_csr  <= 1'b0;
            req_is_dma  <= 1'b0;
            req_csr_addr <= 8'd0;
        end else if (tl_state == TL_IDLE && fsm_acc) begin
            // Capture request
            req_opcode   <= tl_a_opcode;
            req_param    <= tl_a_param[1:0];
            req_size     <= tl_a_size;
            req_source   <= tl_a_source;
            req_wdata    <= tl_a_data;
            req_mask     <= tl_a_mask;
            req_is_read  <= a_is_read;
            req_is_csr   <= a_is_csr;
            req_is_dma   <= a_is_dma;
            req_csr_addr <= a_page[7:0];
        end
    end

    // ── 旁路 FIFO: 压入（DMA 写受理时）/ 弹出（D 握手时）──
    wire fifo_d_valid = ~dfifo_empty && ~fsm_d_valid;   // FSM 的 D 优先
    wire dfifo_pop    = fifo_d_valid && tl_d_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dfifo_wp <= {(DFIFO_AW+1){1'b0}};
            dfifo_rp <= {(DFIFO_AW+1){1'b0}};
        end else begin
            if (dma_wr_acc)
                dfifo_mem[dfifo_wp[DFIFO_AW-1:0]] <= {tl_a_param[1:0], tl_a_size, tl_a_source};
            if (dma_wr_acc)
                dfifo_wp <= dfifo_wp + 1'b1;
            if (dfifo_pop)
                dfifo_rp <= dfifo_rp + 1'b1;
        end
    end

    // ── D 通道输出 mux (FSM 优先, 否则旁路 FIFO 的 AccessAck) ──
    assign tl_d_valid  = fsm_d_valid | fifo_d_valid;
    assign tl_d_opcode = fsm_d_valid ? fsm_d_opcode : TL_OP_ACCESSACK;
    assign tl_d_param  = fsm_d_valid ? fsm_d_param  : dfifo_rdata[7:6];
    assign tl_d_size   = fsm_d_valid ? fsm_d_size   : dfifo_rdata[5:2];
    assign tl_d_source = fsm_d_valid ? fsm_d_source : dfifo_rdata[1:0];
    assign tl_d_sink   = fsm_d_valid ? fsm_d_sink   : 1'b0;
    assign tl_d_data   = fsm_d_valid ? fsm_d_data   : {DATA_WIDTH{1'b0}};
    assign tl_d_error  = fsm_d_valid ? fsm_d_error  : 1'b0;

    // ── CSR Register File write (on accepting PutFullData to CSR space) ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rf_ctrl         <= {DATA_WIDTH{1'b0}};
            rf_status       <= {DATA_WIDTH{1'b0}};
            rf_weight_addr  <= {DATA_WIDTH{1'b0}};
            rf_act_addr     <= {DATA_WIDTH{1'b0}};
            rf_result_addr  <= {DATA_WIDTH{1'b0}};
            rf_num_rows     <= {DATA_WIDTH{1'b0}};
            rf_num_cols     <= {DATA_WIDTH{1'b0}};
            rf_block_size   <= {DATA_WIDTH{1'b0}};
            rf_w_scale      <= {DATA_WIDTH{1'b0}};
            rf_w_bias       <= {DATA_WIDTH{1'b0}};
            rf_w_sup_scale  <= {DATA_WIDTH{1'b0}};
            rf_w_sup_bias   <= {DATA_WIDTH{1'b0}};
        end else begin
            // Write strobe: PutFullData to CSR space accepted via FSM path
            if (fsm_acc && a_is_write && a_is_csr) begin
                case (a_page[7:0])
                    CSR_CTRL:        rf_ctrl        <= tl_a_data;
                    CSR_WEIGHT_ADDR: rf_weight_addr <= tl_a_data;
                    CSR_ACT_ADDR:    rf_act_addr    <= tl_a_data;
                    CSR_RESULT_ADDR: rf_result_addr <= tl_a_data;
                    CSR_NUM_ROWS:    rf_num_rows    <= tl_a_data;
                    CSR_NUM_ROWS_A:  rf_num_rows    <= tl_a_data;
                    CSR_NUM_COLS:    rf_num_cols    <= tl_a_data;
                    CSR_NUM_COLS_A:  rf_num_cols    <= tl_a_data;
                    CSR_BLOCK_SIZE:  rf_block_size  <= tl_a_data;
                    CSR_BLOCK_SIZE_A: rf_block_size <= tl_a_data;
                    CSR_W_SCALE:     rf_w_scale     <= tl_a_data;
                    CSR_W_BIAS:      rf_w_bias      <= tl_a_data;
                    CSR_W_SUP_SCALE: rf_w_sup_scale  <= tl_a_data;
                    CSR_W_SUP_BIAS:  rf_w_sup_bias   <= tl_a_data;
                    default: ;
                endcase
            end

            // STATUS: written by hardware, not by TL
            rf_status <= {56'd0, status_error, status_done, status_busy};

            // CTRL start bit is self-clearing after sampling by controller
            // (controller responsibility; we pass raw bits)
        end
    end

    // ── CSR output assignments ──
    assign csr_start       = rf_ctrl[0];
    assign csr_reset       = rf_ctrl[1];
    assign csr_mode        = rf_ctrl[3:2];
    assign csr_weight_addr = rf_weight_addr[31:0];
    assign csr_act_addr    = rf_act_addr[31:0];
    assign csr_result_addr = rf_result_addr[31:0];
    assign csr_num_rows    = rf_num_rows[15:0];
    assign csr_num_cols    = rf_num_cols[15:0];
    assign csr_block_size  = rf_block_size[15:0];
    assign csr_w_scale     = rf_w_scale[31:0];
    assign csr_w_bias      = rf_w_bias[31:0];
    assign csr_w_sup_scale = rf_w_sup_scale[31:0];
    assign csr_w_sup_bias  = rf_w_sup_bias[31:0];

    // ── DMA State Machine ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_state     <= DMA_IDLE;
            dma_word_cnt  <= 16'd0;
            dma_done      <= 1'b0;
            dma_busy      <= 1'b0;
            dma_done_pulse <= 1'b0;
        end else begin
            dma_done <= 1'b0;  // default: pulse

            case (dma_state)
                DMA_IDLE: begin
                    dma_busy      <= 1'b0;
                    dma_word_cnt  <= 16'd0;
                    dma_done_pulse <= 1'b0;
                    if (dma_start) begin
                        dma_state <= DMA_RUNNING;
                        dma_busy  <= 1'b1;
                    end
                end

                DMA_RUNNING: begin
                    // Count completed DMA transactions (LOAD 写 或 STORE 读)
                    if (dma_count) begin
                        dma_word_cnt <= dma_word_cnt + 16'd1;
                    end

                    // Check for completion (last access accepted this cycle)
                    if ((dma_word_cnt + 16'd1 >= dma_total_words) && dma_count) begin
                        dma_done_pulse <= 1'b1;
                    end

                    // 等旁路 FIFO 的 D 全部排空, 再回报 done (避免丢 AccessAck)
                    if (dma_done_pulse && dfifo_empty) begin
                        dma_done       <= 1'b1;
                        dma_busy       <= 1'b0;
                        dma_state      <= DMA_IDLE;
                        dma_done_pulse <= 1'b0;
                    end
                end

                default: dma_state <= DMA_IDLE;
            endcase
        end
    end

    // ── DMA SRAM interface control ──
    // During DMA_RUNNING, route TL accesses to DMA space to the selected SRAM.
    // LOAD mode (dma_rw=0): PutFullData → SRAM write  (posted, 1 word/cycle)
    // STORE mode (dma_rw=1): Get → SRAM read (data returned via channel D)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_sram_wt_wr_en   <= 1'b0;
            dma_sram_wt_wr_addr <= 16'd0;
            dma_sram_wt_wr_data <= {DATA_WIDTH{1'b0}};
            dma_sram_act_wr_en  <= 1'b0;
            dma_sram_act_wr_addr <= 16'd0;
            dma_sram_act_wr_data <= {DATA_WIDTH{1'b0}};
            dma_sram_acc_rd_en  <= 1'b0;
            dma_sram_acc_rd_addr <= 16'd0;
        end else begin
            // Default: deassert all SRAM controls
            dma_sram_wt_wr_en   <= 1'b0;
            dma_sram_act_wr_en  <= 1'b0;
            dma_sram_acc_rd_en  <= 1'b0;

            // ── LOAD mode: posted write, 1 word/cycle ──
            if (dma_wr_acc) begin
                case (dma_target)
                    2'd0: begin  // Weight SRAM
                        dma_sram_wt_wr_en   <= 1'b1;
                        dma_sram_wt_wr_addr <= dma_word_cnt;
                        dma_sram_wt_wr_data <= tl_a_data;
                    end
                    2'd1: begin  // Activation SRAM
                        dma_sram_act_wr_en  <= 1'b1;
                        dma_sram_act_wr_addr <= dma_word_cnt;
                        dma_sram_act_wr_data <= tl_a_data;
                    end
                    2'd2: begin  // Accumulator SRAM write (unused in LOAD)
                        // no SRAM write
                    end
                    default: ;
                endcase
            end

            // ── STORE mode: host reads from DMA window ← SRAM (FSM 路径) ──
            if (dma_rd_acc) begin
                case (dma_target)
                    2'd2: begin  // Accumulator SRAM
                        dma_sram_acc_rd_en  <= 1'b1;
                        dma_sram_acc_rd_addr <= dma_word_cnt;
                    end
                    default: ; // Weight/Act SRAM reads not used in STORE
                endcase
            end
        end
    end

endmodule
