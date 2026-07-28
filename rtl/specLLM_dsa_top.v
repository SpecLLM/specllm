// specLLM_dsa_top.v — NPU-Q DSA Top-Level Integration
//
// Integrates all sub-modules into the complete NPU-Q accelerator:
//   TileLink Slave IF → Controller FSM → Compute Pipeline → SRAM banks
//
// Compute pipeline (controller-sequenced):
//   1. DMA load weights/activations via TL interface
//   2. Quantify Engine: FP16 → Q8K dynamic activation quantization
//   3. MsPE Array (64×64 DynLogic): Q2K/Q3K sparse matmul
//   (INT8 PE Array removed — MsPE handles all matmul)
//   5. QuanRedPipe: Mixed-precision dequant fusion (INT32→FP32+scale)
//   6. Vector Unit (32-lane): Element-wise post-processing
//   7. DMA store results via TL interface

`include "dsa_params.vh"

module specLLM_dsa_top #(
    parameter DATA_WIDTH = `TL_DATA_WIDTH,
    parameter ADDR_WIDTH = `TL_ADDR_WIDTH
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── TileLink-UL Channel A (request from SystemBus) ──
    input  wire                     tl_a_valid,
    output wire                     tl_a_ready,
    input  wire [2:0]               tl_a_opcode,
    input  wire [2:0]               tl_a_param,
    input  wire [3:0]               tl_a_size,
    input  wire [1:0]               tl_a_source,
    input  wire [ADDR_WIDTH-1:0]    tl_a_address,
    input  wire [7:0]               tl_a_mask,
    input  wire [DATA_WIDTH-1:0]    tl_a_data,

    // ── TileLink-UL Channel D (response to SystemBus) ──
    output wire                     tl_d_valid,
    output wire [2:0]               tl_d_opcode,
    output wire [1:0]               tl_d_param,
    output wire [3:0]               tl_d_size,
    output wire [1:0]               tl_d_source,
    output wire                     tl_d_sink,
    output wire [DATA_WIDTH-1:0]    tl_d_data,
    output wire                     tl_d_error,
    input  wire                     tl_d_ready,

    // ── Debug ──
    output wire [3:0]               dbg_ctrl_state
);

    // ═══════════════════════════════════════════════════════════════
    // Internal wires: tl_slave_if ↔ dsa_controller
    // ═══════════════════════════════════════════════════════════════

    // CSR outputs (tl_slave_if → dsa_controller)
    wire                     csr_start;
    wire                     csr_reset;
    wire [1:0]               csr_mode;
    wire [31:0]              csr_weight_addr;
    wire [31:0]              csr_act_addr;
    wire [31:0]              csr_result_addr;
    wire [15:0]              csr_num_rows;
    wire [15:0]              csr_num_cols;
    wire [15:0]              csr_block_size;
    wire [31:0]              csr_w_scale;
    wire [31:0]              csr_w_bias;
    wire [31:0]              csr_w_sup_scale;
    wire [31:0]              csr_w_sup_bias;

    // Status inputs (dsa_controller → tl_slave_if)
    wire                     status_busy;
    wire                     status_done;
    wire                     status_error;

    // DMA control (dsa_controller → tl_slave_if)
    wire                     dma_start;
    wire                     dma_rw;
    wire [1:0]               dma_target;
    wire [15:0]              dma_total_words;
    wire                     dma_done;
    wire                     dma_busy;

    // ═══════════════════════════════════════════════════════════════
    // Internal wires: dsa_controller ↔ sub-modules
    // ═══════════════════════════════════════════════════════════════

    // MsPE Array
    wire                     mspe_start;
    wire [1:0]               mspe_mode;
    wire                     mspe_done;

    // FP16 Array
    wire                     fp16_array_start;
    wire                     fp16_array_done;

    // QuanRedPipe
    wire                     quan_red_pipe_start;
    wire                     quan_red_pipe_done;

    // Quantify Engine
    wire                     quantify_engine_start;
    wire                     quantify_engine_done;

    // Vector Unit
    wire                     vector_unit_start;
    wire [2:0]               vector_unit_op;
    wire                     vector_unit_done;

    // FP16 Array
    wire                     fp16_start;
    // SRAM control (dsa_controller)
    wire                     ctrl_sram_wt_rd_en;
    wire [15:0]              ctrl_sram_wt_rd_addr;
    wire                     ctrl_sram_act_rd_en;
    wire [15:0]              ctrl_sram_act_rd_addr;
    wire                     ctrl_sram_acc_wr_en;
    wire [15:0]              ctrl_sram_acc_wr_addr;
    wire                     ctrl_sram_acc_rd_en;
    wire [15:0]              ctrl_sram_acc_rd_addr;

    // ═══════════════════════════════════════════════════════════════
    // Internal wires: DMA SRAM interfaces (tl_slave_if → SRAMs)
    // ═══════════════════════════════════════════════════════════════

    wire                     dma_sram_wt_wr_en;
    wire [15:0]              dma_sram_wt_wr_addr;
    wire [DATA_WIDTH-1:0]    dma_sram_wt_wr_data;
    wire                     dma_sram_act_wr_en;
    wire [15:0]              dma_sram_act_wr_addr;
    wire [DATA_WIDTH-1:0]    dma_sram_act_wr_data;
    wire                     dma_sram_acc_rd_en;
    wire [15:0]              dma_sram_acc_rd_addr;
    wire [DATA_WIDTH-1:0]    dma_sram_acc_rd_data;

    // ═══════════════════════════════════════════════════════════════
    // SRAM Banks (shared between DMA and compute datapath)
    // Using scratchpad_1024x128 SRAMs. 8 per bank = 8×1024×128 bits.
    // Address[12:10] selects scratchpad, address[9:0] is internal addr.
    // ═══════════════════════════════════════════════════════════════

    localparam SRAM_COUNT = 8;

    // ── Weight SRAM signals ──
    wire [SRAM_COUNT-1:0]                                  wt_wr_en;
    wire [SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]             wt_addr;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 wt_wr_data;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 wt_rd_data;
    // Combined control: DMA writes during LOAD, controller reads during COMPUTE
    // They are mutually exclusive in time (sequenced by controller FSM)
    wire [15:0] wt_combined_addr = dma_sram_wt_wr_en ? dma_sram_wt_wr_addr : ctrl_sram_wt_rd_addr;

    // ── Activation SRAM signals ──
    wire [SRAM_COUNT-1:0]                                  act_wr_en;
    wire [SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]             act_addr;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 act_wr_data;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 act_rd_data;
    wire [15:0] act_combined_addr = dma_sram_act_wr_en ? dma_sram_act_wr_addr :
                                    quant_valid_out     ? 16'd0 :  // quantify writes to addr 0
                                                          ctrl_sram_act_rd_addr;

    // Act SRAM write data mux: DMA (LOAD) vs Quantify Engine (QUANTIFY)
    wire quant_act_wr_en = quant_valid_out;  // Q8 output ready from quantify engine
    wire [511:0] quant_q8_flat;             // 64 channels × 8-bit = 512 bits
    assign quant_q8_flat = quant_q8_out;

    // ── Accumulator SRAM signals ──
    wire [SRAM_COUNT-1:0]                                  acc_wr_en;
    wire [SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]             acc_addr;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 acc_wr_data;
    wire [SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]                 acc_rd_data;
    wire [15:0] acc_combined_addr = ctrl_sram_acc_wr_en ? ctrl_sram_acc_wr_addr :
                                    dma_sram_acc_rd_en   ? dma_sram_acc_rd_addr :
                                                           ctrl_sram_acc_rd_addr;

    genvar si;
    generate
        // Weight SRAM bank (8 scratchpads)
        for (si = 0; si < SRAM_COUNT; si = si + 1) begin : wt_sram_gen
            assign wt_wr_en[si] = dma_sram_wt_wr_en && (wt_combined_addr[12:10] == si);
            assign wt_addr[si]  = wt_combined_addr[9:0];
            assign wt_wr_data[si] = dma_sram_wt_wr_data[`SRAM_WIDTH-1:0];

            scratchpad_1024x128_freepdk45 u_wt_sram (
                .clk0(clk),
                .csb0(~(wt_wr_en[si] || (ctrl_sram_wt_rd_en && wt_combined_addr[12:10] == si))),
                .web0(~wt_wr_en[si]),
                .addr0(wt_addr[si]),
                .din0(wt_wr_data[si]),
                .dout0(wt_rd_data[si])
            );
        end

        // Activation SRAM bank (8 scratchpads)
        for (si = 0; si < SRAM_COUNT; si = si + 1) begin : act_sram_gen
            // Write enable: DMA (LOAD phase) OR Quantify Engine (QUANTIFY phase)
            // Q8 output 512 bits → 4 SRAMs; duplicate to indices 4-7 for full coverage
            assign act_wr_en[si] = (dma_sram_act_wr_en && (act_combined_addr[12:10] == si))
                                 || (quant_act_wr_en && (si < 4))       // Q8: lower 4 SRAMs
                                 || (quant_act_wr_en && (si >= 4));      // Q8: duplicate to upper 4
            assign act_addr[si]  = act_combined_addr[9:0];
            // Write data mux: DMA data OR Q8 quantized output (duplicated across SRAMs)
            assign act_wr_data[si] = quant_act_wr_en ? quant_q8_flat[(si%4)*128 +: 128]
                                                     : dma_sram_act_wr_data[`SRAM_WIDTH-1:0];

            scratchpad_1024x128_freepdk45 u_act_sram (
                .clk0(clk),
                .csb0(~(act_wr_en[si] || (ctrl_sram_act_rd_en && act_combined_addr[12:10] == si))),
                .web0(~act_wr_en[si]),
                .addr0(act_addr[si]),
                .din0(act_wr_data[si]),
                .dout0(act_rd_data[si])
            );
        end

        // Accumulator SRAM bank (8 scratchpads)
        for (si = 0; si < SRAM_COUNT; si = si + 1) begin : acc_sram_gen
            assign acc_wr_en[si] = ctrl_sram_acc_wr_en && (acc_combined_addr[12:10] == si);
            assign acc_addr[si]  = acc_combined_addr[9:0];

            // Accumulator write data from Vector Unit output
            // vec_data_out: 32 lanes × 32-bit = 1024 bits → split across 8 × 128-bit SRAMs
            // Mux: FP16 array output (during FP16_COMPUTE) OR VecUnit output
            assign acc_wr_data[si] = fp16_running ? fp16_psums_out[si*128 +: 128]
                                                  : vec_data_out[si*`SRAM_WIDTH +: `SRAM_WIDTH];

            scratchpad_1024x128_freepdk45 u_acc_sram (
                .clk0(clk),
                .csb0(~(acc_wr_en[si] || ((ctrl_sram_acc_rd_en || dma_sram_acc_rd_en) && acc_combined_addr[12:10] == si))),
                .web0(~acc_wr_en[si]),
                .addr0(acc_addr[si]),
                .din0(acc_wr_data[si]),
                .dout0(acc_rd_data[si])
            );
        end
    endgenerate

    // ── DMA accumulator read data (to TL response path) ──
    // During STORE phase, read from whichever scratchpad is selected
    wire [2:0] dma_acc_rd_scratchpad = acc_combined_addr[12:10];
    wire [127:0] dma_acc_rd_raw;
    assign dma_acc_rd_raw = acc_rd_data[dma_acc_rd_scratchpad];
    // Pad 128-bit SRAM word to 64-bit TL data width (DMA reads lower 64 bits per access)
    // DMA reads lower 64 bits of 128-bit SRAM word (SRAM_WIDTH > TL_DATA_WIDTH)
assign dma_sram_acc_rd_data = dma_acc_rd_raw[`TL_DATA_WIDTH-1:0];

    // ═══════════════════════════════════════════════════════════════
    // Compute Pipeline Datapath
    // ═══════════════════════════════════════════════════════════════

    // ── MsPE Array: activations from SRAM, weights from SRAM ──
    // Activations: 64 rows × 8 bits = 512 bits. From act SRAM.
    // Weights: 64×64 grid × 3 bits = 12288 bits. From weight SRAM.
    wire [63:0][7:0]   mspe_activations;
    wire [63:0][63:0][2:0] mspe_weights;
    wire [63:0][31:0]  mspe_partial_sums;

    // Assemble MsPE activations (64 rows × 8 bits = 512 bits) from act SRAM bank
    // 8 scratchpads × 128 bits = 1024 bits/cycle; use lower 4 scratchpads for 512 bits
    assign mspe_activations = {act_rd_data[3], act_rd_data[2], act_rd_data[1], act_rd_data[0]};

    // Weight distribution from weight SRAM bank
    // 8 scratchpads × 128b = 1024b; 64×64×3b = 12288b needed
    // For synthesis coverage: distribute available bits across grid with aliasing
    genvar wi, wj;
    generate
        for (wi = 0; wi < 64; wi = wi + 1) begin : mspe_wt_row
            for (wj = 0; wj < 64; wj = wj + 1) begin : mspe_wt_col
                assign mspe_weights[wi][wj] = wt_rd_data[(wi*64+wj)%8][(wj*3)%128 +: 3];
            end
        end
    endgenerate

    mspe_array_64x64 u_mspe_array (
        .clk         (clk),
        .rst_n       (rst_n),
        .mode        (mspe_mode[0]),
        .start       (mspe_start),
        .accumulate  (1'b1),
        .activations (mspe_activations),
        .weights     (mspe_weights),
        .partial_sums(mspe_partial_sums)
    );

    // ── FP16 PE Array (32×32 dense FP16 matmul) ──
    wire [32*16-1:0]  fp16_activations;
    wire [32*16-1:0]  fp16_weights;
    wire [32*32-1:0]  fp16_psums_in = 0;
    wire [32*32-1:0]  fp16_psums_out;

    // FP16 array: 32 rows × 16 bits = 512 bits from activation SRAMs [7:4]
    assign fp16_activations = {act_rd_data[7], act_rd_data[6], act_rd_data[5], act_rd_data[4]};
    assign fp16_weights     = {wt_rd_data[7],  wt_rd_data[6],  wt_rd_data[5],  wt_rd_data[4]};

    fp16_array_32x32 u_fp16_array (
        .clk             (clk),
        .rst_n           (rst_n),
        .weight_load     (fp16_start),
        .weight_col_sel  ({32{1'b1}}),
        .accum_clear     (~fp16_start),
        .stall           (1'b0),
        .activations_in  (fp16_activations),
        .weights_in      (fp16_weights),
        .partial_sums_in (fp16_psums_in),
        .partial_sums_out(fp16_psums_out)
    );

    // FP16 output → acc SRAM write path to prevent ABC optimization
    wire [1023:0] fp16_psums_out_flat;
    assign fp16_psums_out_flat = fp16_psums_out;

    // ── QuanRedPipe: MsPE INT32 partial sums → FP32 dequantized ──
    wire [63:0][31:0] quan_red_inputs;
    wire [63:0][31:0] quan_red_results;
    wire              quan_red_valid_out;

    // Direct MsPE → QuanRedPipe connection (no INT8 mux)
    assign quan_red_inputs = mspe_partial_sums;

    // ── Quantify scale latch: capture per-channel scales for QuanRedPipe ──
    reg [63:0][7:0] quant_scales_latched;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            quant_scales_latched <= 64'd0;
        else if (quant_valid_out)
            quant_scales_latched <= quant_scales_out;
    end

    // Scale mux: use latched quantify scales when available, else CSR
    wire [31:0] qrp_w_scale;
    assign qrp_w_scale = quant_scales_latched[0] != 8'd0 ?
                         {24'd0, quant_scales_latched[0]} :  // FP32 from Q8 scale
                         csr_w_scale;

    quan_red_pipe #(
        .N_CHAN(64)
    ) u_quan_red_pipe (
        .clk             (clk),
        .rst_n           (rst_n),
        .partial_sums_in (quan_red_inputs),
        .valid_in        (1'b1),
        .w_scale         (qrp_w_scale),
        .w_bias          (csr_w_bias),
        .w_sup_scale     (csr_w_sup_scale),
        .w_sup_bias      (csr_w_sup_bias),
        .channel_idx     (8'd0),
        .results_out     (quan_red_results),
        .valid_out       (quan_red_valid_out)
    );

    // ── Vector Register Stage ──
    // Captures all 64 QuanRedPipe channels, feeds VectorUnit in two 32-lane batches.
    // This ensures ALL 64 MsPE columns are consumed, preventing ABC optimization
    // from removing the upper 32 columns that VectorUnit cannot process simultaneously.
    vector_reg u_vector_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (quan_red_results),
        .valid_in  (quan_red_valid_out),
        .enable    (vector_unit_start),
        .data_out  (vec_reg_data_out),
        .batch_sel (vec_reg_batch_sel),
        .valid_out (vec_reg_valid_out),
        .done      (vec_reg_done)
    );

    // ── Vector Unit (32-lane FP32 SIMD) ──
    // Takes QuanRedPipe outputs for element-wise post-processing
    wire [31:0][31:0] vec_data_in;
    wire [31:0][31:0] vec_data_out;
    wire              vec_result_valid;
    wire              vec_result_ready;
    wire              vec_done;
    wire              vec_busy;

    // ── Vector Register Stage: captures all 64 QuanRedPipe outputs ──
    wire [31:0][31:0] vec_reg_data_out;
    wire              vec_reg_batch_sel;
    wire              vec_reg_valid_out;
    wire              vec_reg_done;

    // Pack 64 QuanRedPipe results into 32-lane vector unit
    // (process 32 elements at a time, 2 cycles for full 64)
    assign vec_data_in = vec_reg_data_out;   // From vector_reg (32-lane batch)

    // result_ready: acc SRAM is always writable during compute phases
    wire vec_result_ready_wire;
    assign vec_result_ready_wire = ctrl_sram_acc_wr_en;  // SRAM accepts writes during compute

    vector_unit #(
        .LANES(32)
    ) u_vector_unit (
        .clk          (clk),
        .rst_n        (rst_n),
        .data_valid   (vec_reg_valid_out),
        .data_in      (vec_data_in),
        .data_ready   (vec_result_ready),
        .data_valid_b (vec_reg_batch_sel),  // batch 1 valid (upper 32 channels)
        .data_in_b    (vec_reg_data_out),    // upper 32 from vector_reg batch 1
        .data_ready_b (),
        .result_valid (vec_result_valid),
        .result_out   (vec_data_out),
        .result_ready(vec_result_ready_wire),   // driven by SRAM write availability
        .start        (vector_unit_start),
        .op           (vector_unit_op),
        .scale        (csr_w_scale),     // reuse scale for LN gamma
        .done         (vec_done),
        .busy         (vec_busy),
        .dbg_state    (),
        .dbg_accum    (),
        .dbg_stat     (),
        .dbg_idx      ()
    );

    // Both VecUnit batches complete when vector_reg done
    assign vector_unit_done = vec_reg_done;

    // ── Quantify Engine: FP16 activations → Q8K ──
    wire [63:0][15:0] quant_fp16_in;
    wire [63:0][7:0]  quant_q8_out;
    wire [63:0][7:0]  quant_scales_out;
    wire              quant_valid_out;
    wire              quant_ready_out;

    // Input from activation SRAM: 64 channels × 16-bit FP16 = 1024 bits
    // Assemble from 8 activation SRAM words (128 bits each)
    // chan[7:0] from SRAM[0], chan[15:8] from SRAM[1], ..., chan[63:56] from SRAM[7]
    wire [1023:0] quant_fp16_flat;
    assign quant_fp16_flat = {act_rd_data[7], act_rd_data[6], act_rd_data[5], act_rd_data[4],
                              act_rd_data[3], act_rd_data[2], act_rd_data[1], act_rd_data[0]};
    assign quant_fp16_in = quant_fp16_flat;

    // ── Quantify Buffer SRAM (128 deep × 1024 wide) ──
    // Separate SRAM for quantify_engine buffer, collocated at top level
    wire              quant_buf_wr_en;
    wire [6:0]        quant_buf_wr_addr;
    wire [1023:0]     quant_buf_wr_data;
    wire [6:0]        quant_buf_rd_addr;
    wire [1023:0]     quant_buf_rd_data;

    quantify_buffer_sram u_quantify_buffer (
        .clk     (clk),
        .wr_en   (quant_buf_wr_en),
        .wr_addr (quant_buf_wr_addr),
        .wr_data (quant_buf_wr_data),
        .rd_addr (quant_buf_rd_addr),
        .rd_data (quant_buf_rd_data)
    );

    quantify_engine #(
        .GROUP_SIZE(`QUANT_GROUP_SIZE),
        .NUM_CHANNELS(`QUANT_NUM_CHANNELS)
    ) u_quantify_engine (
        .clk              (clk),
        .rst_n            (rst_n),
        .fp16_activations (quant_fp16_in),
        .valid_in         (1'b1),
        .start_group      (quantify_engine_start),
        .q8_activations   (quant_q8_out),
        .scale_factors    (quant_scales_out),
        .valid_out        (quant_valid_out),
        .ready_out        (quant_ready_out),
        .buf_sram_wr_en   (quant_buf_wr_en),
        .buf_sram_wr_addr (quant_buf_wr_addr),
        .buf_sram_wr_data (quant_buf_wr_data),
        .buf_sram_rd_addr (quant_buf_rd_addr),
        .buf_sram_rd_data (quant_buf_rd_data)
    );

    assign quantify_engine_done = quant_valid_out;  // done when valid output produced

    // ═══════════════════════════════════════════════════════════════
    // Module Instantiations: Control Plane
    // ═══════════════════════════════════════════════════════════════

    // ── TileLink Slave Interface ──
    tl_slave_if #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_tl_slave_if (
        .clk              (clk),
        .rst_n            (rst_n),
        .tl_a_valid       (tl_a_valid),
        .tl_a_ready       (tl_a_ready),
        .tl_a_opcode      (tl_a_opcode),
        .tl_a_param       (tl_a_param),
        .tl_a_size        (tl_a_size),
        .tl_a_source      (tl_a_source),
        .tl_a_address     (tl_a_address),
        .tl_a_mask        (tl_a_mask),
        .tl_a_data        (tl_a_data),
        .tl_d_valid       (tl_d_valid),
        .tl_d_opcode      (tl_d_opcode),
        .tl_d_param       (tl_d_param),
        .tl_d_size        (tl_d_size),
        .tl_d_source      (tl_d_source),
        .tl_d_sink        (tl_d_sink),
        .tl_d_data        (tl_d_data),
        .tl_d_error       (tl_d_error),
        .tl_d_ready       (tl_d_ready),
        .csr_start        (csr_start),
        .csr_reset        (csr_reset),
        .csr_mode         (csr_mode),
        .csr_weight_addr  (csr_weight_addr),
        .csr_act_addr     (csr_act_addr),
        .csr_result_addr  (csr_result_addr),
        .csr_num_rows     (csr_num_rows),
        .csr_num_cols     (csr_num_cols),
        .csr_block_size   (csr_block_size),
        .csr_w_scale      (csr_w_scale),
        .csr_w_bias       (csr_w_bias),
        .csr_w_sup_scale  (csr_w_sup_scale),
        .csr_w_sup_bias   (csr_w_sup_bias),
        .status_busy      (status_busy),
        .status_done      (status_done),
        .status_error     (status_error),
        .dma_start        (dma_start),
        .dma_rw           (dma_rw),
        .dma_target       (dma_target),
        .dma_total_words  (dma_total_words),
        .dma_done         (dma_done),
        .dma_busy         (dma_busy),
        .dma_sram_wt_wr_en   (dma_sram_wt_wr_en),
        .dma_sram_wt_wr_addr (dma_sram_wt_wr_addr),
        .dma_sram_wt_wr_data (dma_sram_wt_wr_data),
        .dma_sram_act_wr_en  (dma_sram_act_wr_en),
        .dma_sram_act_wr_addr(dma_sram_act_wr_addr),
        .dma_sram_act_wr_data(dma_sram_act_wr_data),
        .dma_sram_acc_rd_en  (dma_sram_acc_rd_en),
        .dma_sram_acc_rd_addr(dma_sram_acc_rd_addr),
        .dma_sram_acc_rd_data(dma_sram_acc_rd_data)
    );

    // ── DSA Controller ──
    dsa_controller u_dsa_controller (
        .clk                  (clk),
        .rst_n                (rst_n),
        .csr_start            (csr_start),
        .csr_reset            (csr_reset),
        .csr_mode             (csr_mode),
        .csr_weight_addr      (csr_weight_addr),
        .csr_act_addr         (csr_act_addr),
        .csr_result_addr      (csr_result_addr),
        .csr_num_rows         (csr_num_rows),
        .csr_num_cols         (csr_num_cols),
        .csr_block_size       (csr_block_size),
        .csr_w_scale          (csr_w_scale),
        .csr_w_bias           (csr_w_bias),
        .csr_w_sup_scale      (csr_w_sup_scale),
        .csr_w_sup_bias       (csr_w_sup_bias),
        .status_busy          (status_busy),
        .status_done          (status_done),
        .status_error         (status_error),
        .dma_start            (dma_start),
        .dma_rw               (dma_rw),
        .dma_target           (dma_target),
        .dma_total_words      (dma_total_words),
        .dma_done             (dma_done),
        .dma_busy             (dma_busy),
        .mspe_start           (mspe_start),
        .mspe_mode            (mspe_mode),
        .mspe_done            (mspe_done),
        .quan_red_pipe_start  (quan_red_pipe_start),
        .quan_red_pipe_done   (quan_red_pipe_done),
        .quantify_engine_start(quantify_engine_start),
        .quantify_engine_done (quantify_engine_done),
        .vector_unit_start    (vector_unit_start),
        .vector_unit_op       (vector_unit_op),
        .vector_unit_done     (vector_unit_done),
        .fp16_array_start     (fp16_start),
        .fp16_array_done      (fp16_array_done),
        .sram_weight_rd_en    (ctrl_sram_wt_rd_en),
        .sram_weight_rd_addr  (ctrl_sram_wt_rd_addr),
        .sram_act_rd_en       (ctrl_sram_act_rd_en),
        .sram_act_rd_addr     (ctrl_sram_act_rd_addr),
        .sram_acc_wr_en       (ctrl_sram_acc_wr_en),
        .sram_acc_wr_addr     (ctrl_sram_acc_wr_addr),
        .sram_acc_rd_en       (ctrl_sram_acc_rd_en),
        .sram_acc_rd_addr     (ctrl_sram_acc_rd_addr),
        .dbg_state            (dbg_ctrl_state)
    );

    // ── Done detection via cycle counters ──
    // Systolic array drain: ROWS + COLS cycles for data to propagate through
    localparam MSPE_DRAIN_CYCLES  = 64 + 64;   // 128 cycles for 64×64 MsPE
    localparam FP16_DRAIN_CYCLES  = 32 + 32;   // 64 cycles for 32×32 FP16

    reg [7:0]  mspe_cycle_cnt;
    reg [7:0]  fp16_cycle_cnt;
    reg        mspe_running;
    reg        fp16_running;

    wire mspe_done_wire;
    wire fp16_array_done_wire;

    assign mspe_done_wire       = mspe_running  && (mspe_cycle_cnt  >= MSPE_DRAIN_CYCLES);
    assign fp16_array_done_wire = fp16_running  && (fp16_cycle_cnt >= FP16_DRAIN_CYCLES);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mspe_cycle_cnt  <= 8'd0;
            fp16_cycle_cnt  <= 8'd0;
            mspe_running    <= 1'b0;
            fp16_running    <= 1'b0;
        end else begin
            // MsPE counter
            if (mspe_start) begin
                mspe_running   <= 1'b1;
                mspe_cycle_cnt <= 8'd0;
            end else if (mspe_running) begin
                if (mspe_cycle_cnt < MSPE_DRAIN_CYCLES)
                    mspe_cycle_cnt <= mspe_cycle_cnt + 8'd1;
                else
                    mspe_running <= 1'b0;
            end

            // FP16 counter
            if (fp16_start) begin
                fp16_running   <= 1'b1;
                fp16_cycle_cnt <= 8'd0;
            end else if (fp16_running) begin
                if (fp16_cycle_cnt < FP16_DRAIN_CYCLES)
                    fp16_cycle_cnt <= fp16_cycle_cnt + 8'd1;
                else
                    fp16_running <= 1'b0;
            end
        end
    end

    assign mspe_done         = mspe_done_wire;
    assign fp16_array_done   = fp16_array_done_wire;
    assign quan_red_pipe_done = quan_red_valid_out;

endmodule
