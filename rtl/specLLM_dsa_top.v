// specLLM_dsa_top.v — NPU-Q DSA Top-Level Integration
//
// Integrates all sub-modules into the complete NPU-Q accelerator:
//   TileLink Slave IF → Controller FSM → Compute Pipeline → SRAM banks
//
// Compute pipeline (controller-sequenced):
//   1. DMA load weights/activations via TL interface
//   2. Quantify Engine: FP16 → Q8K dynamic activation quantization
//   3. MsPE Array (32×32 DynLogic): Q2K(4-lane)/Q3K(2-lane) sparse matmul
//   (INT8 PE Array removed — MsPE handles all matmul)
//   5. QuanRedPipe (N_CHAN=32): Mixed-precision dequant fusion (INT32→FP32+scale)
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

    // ── Master DMA (2026-10-10): DSA-initiated memory access, connects to SystemBus fbus ──
    input  wire                     m_dma_start,
    input  wire                     m_dma_rw,      // 0 = read DRAM, 1 = write DRAM
    input  wire [31:0]              m_dma_base_addr,
    input  wire [15:0]              m_dma_n_words,
    output wire                     m_dma_busy,
    output wire                     m_dma_done,
    output wire [63:0]              m_dma_checksum,
    // TL-UL master port
    output wire                     m_a_valid,
    input  wire                     m_a_ready,
    output wire [2:0]               m_a_opcode,
    output wire [3:0]               m_a_size,
    output wire [4:0]               m_a_source,
    output wire [31:0]              m_a_address,
    output wire [63:0]              m_a_data,
    input  wire                     m_d_valid,
    output wire                     m_d_ready,
    input  wire [2:0]               m_d_opcode,
    input  wire [4:0]               m_d_source,
    input  wire [63:0]              m_d_data,

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
    // MsPE weight-load handshake (driven by dsa_controller; see the weight-load FSM)
    wire                     mspe_wt_load_req;    // asserted before COMPUTE to start 8 wide reads
    wire [15:0]              mspe_wt_base_addr;   // first wt SRAM row of this tile
    wire                     mspe_wt_load_busy;   // loading in progress, reported by the array
    wire [2:0]               mspe_wt_load_idx;    // array output: word index to read this cycle (0..7)
    wire [1023:0]            mspe_wt_wide;        // wide word assembled from 8 scratchpads
    // Driven by dsa_controller state S_MSPS_WLOAD (QUANTIFY -> MSPS_WLOAD -> COMPUTE_MSPS)

    // FP16 Array
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
    // Forward declarations (iverilog requires declaration-before-use; semantics unchanged)
    // ═══════════════════════════════════════════════════════════════
    wire [32*32-1:0]         fp16_psums_out;
    reg                      fp16_running;
    wire [31:0][31:0]        vec_data_out;
    wire                     vec_result_ready;
    wire [31:0][31:0]        vec_reg_data_out;
    wire                     vec_reg_batch_sel;
    wire                     vec_reg_valid_out;
    wire                     vec_reg_done;
    wire [32*32-1:0]         vec_data_out_flat = vec_data_out;   // iverilog: flatten for variable indexing
    wire [15:0][7:0]         quant_q8_out;
    wire [15:0][7:0]         quant_scales_out;
    wire                     quant_valid_out;

    // ═══════════════════════════════════════════════════════════════
    // SRAM Banks (shared between DMA and compute datapath)
    // Using scratchpad_1024x128 SRAMs (1024 words × 128 b = 16 KB each).
    //   addr[9:0]    internal address (1024 words)
    //   addr[12:10]  selects a scratchpad within the block (8 = one 1024-bit word)
    //   addr[13]     acc only: selects the ping-pong buffer (0 = group A, 1 = group B)
    //
    //   Capacity: wt 8x16 KB = 128 KB / act 8x16 KB = 128 KB / acc 16x16 KB = 256 KB
    // ═══════════════════════════════════════════════════════════════

    // The three banks no longer have the same macro count (legacy used one SRAM_COUNT=8)
    localparam WT_SRAM_COUNT  = 8;      // weight  128 KB
    localparam ACT_SRAM_COUNT = 8;      // activation 128 KB
    localparam ACC_SRAM_COUNT = 16;     // accumulator 256 KB = 2 groups x 8 (ping-pong)
    localparam ACC_GROUP      = 8;      // 8 scratchpads per group

    // ── Weight SRAM signals ──
    wire [WT_SRAM_COUNT-1:0]                               wt_wr_en;
    wire [WT_SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]          wt_addr;
    wire [WT_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]              wt_wr_data;
    wire [WT_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]              wt_rd_data;
    // Combined control: DMA writes during LOAD, controller reads during COMPUTE
    // They are mutually exclusive in time (sequenced by controller FSM)
    // During weight load: the address is formed from the array's wt_load_idx (0..7); see mspe_array_32x32
    wire [15:0] wt_load_sram_addr = {mspe_wt_base_addr[15:3], mspe_wt_load_idx};
    wire [15:0] wt_combined_addr = dma_sram_wt_wr_en ? dma_sram_wt_wr_addr
                                 : mspe_wt_load_busy ? wt_load_sram_addr      // override during load
                                                     : ctrl_sram_wt_rd_addr;

    // ── Activation SRAM signals ──
    wire [ACT_SRAM_COUNT-1:0]                              act_wr_en;
    wire [ACT_SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]         act_addr;
    wire [ACT_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]             act_wr_data;
    wire [ACT_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]             act_rd_data;
    wire [15:0] act_combined_addr = dma_sram_act_wr_en ? dma_sram_act_wr_addr :
                                    quant_valid_out     ? 16'd0 :  // quantify writes to addr 0
                                                          ctrl_sram_act_rd_addr;

    // Act SRAM write data mux: DMA (LOAD) vs Quantify Engine (QUANTIFY)
    wire quant_act_wr_en = quant_valid_out;  // Q8 output ready from quantify engine
    wire [127:0] quant_q8_flat;             // 16 channels × 8-bit = 128 bits
    assign quant_q8_flat = quant_q8_out;

    // ── Accumulator SRAM signals (16 = 2 groups x 8, ping-pong) ──
    wire [ACC_SRAM_COUNT-1:0]                              acc_wr_en;
    wire [ACC_SRAM_COUNT-1:0][`SRAM_ADDR_BITS-1:0]         acc_addr;
    wire [ACC_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]             acc_wr_data;
    wire [ACC_SRAM_COUNT-1:0][`SRAM_WIDTH-1:0]             acc_rd_data;
    wire [ACC_SRAM_COUNT*`SRAM_WIDTH-1:0]                  acc_rd_flat = acc_rd_data;   // iverilog: flatten for variable indexing
    wire [15:0] acc_combined_addr = ctrl_sram_acc_wr_en ? ctrl_sram_acc_wr_addr :
                                    dma_sram_acc_rd_en   ? dma_sram_acc_rd_addr :
                                                           ctrl_sram_acc_rd_addr;
    // ── ping-pong buffer select (acc only) ──
    //    write group = addr[13], read group = addr[13] (carried in DMA/controller addresses)
    wire        acc_wr_group = acc_combined_addr[13];
    wire        acc_rd_group = acc_combined_addr[13];

    genvar si;
    generate
        // ── Weight SRAM bank (8 × 16 KB = 128 KB) ──
        // csb fix: during a read all 8 scratchpads are enabled (shared addr[9:0] to form
        //   a 1024-bit word); only writes select one via addr[12:10].
        //   The earlier read enable was `rd_en && addr[12:10]==si`, so only one dout was
        //   valid and the other seven stayed X -- the root cause of the broken weight path.
        for (si = 0; si < WT_SRAM_COUNT; si = si + 1) begin : wt_sram_gen
            assign wt_wr_en[si] = dma_sram_wt_wr_en && (wt_combined_addr[12:10] == si);
            assign wt_addr[si]  = wt_combined_addr[9:0];
            assign wt_wr_data[si] = dma_sram_wt_wr_data;   // 64b -> 128b zero-extend (earlier [...127:0] out-of-range select produced X)

            scratchpad_1024x128_freepdk45 u_wt_sram (
                .clk0(clk),
                .csb0(~(ctrl_sram_wt_rd_en || mspe_wt_load_busy || wt_wr_en[si])),
                .web0(~wt_wr_en[si]),
                .addr0(wt_addr[si]),
                .din0(wt_wr_data[si]),
                .dout0(wt_rd_data[si])
            );
        end

        // ── Activation SRAM bank (8 × 16 KB = 128 KB) ──
        // csb fix as above: all 8 enabled during reads.
        // Two earlier issues are recorded here (not changed in this pass; see TODO at end of file):
        //   (a) `quant_act_wr_en` writes the same Q8 into all 8 -> 7 redundant copies;
        //   (b) act[2]/[3] dout0 has no consumer (kept alive by (* keep *)).
        for (si = 0; si < ACT_SRAM_COUNT; si = si + 1) begin : act_sram_gen
            // Write: DMA (LOAD) selects one via addr[12:10]; Quantify (QUANTIFY) still writes all
            assign act_wr_en[si] = (dma_sram_act_wr_en && (act_combined_addr[12:10] == si))
                                 || quant_act_wr_en;
            assign act_addr[si]  = act_combined_addr[9:0];
            assign act_wr_data[si] = quant_act_wr_en ? quant_q8_flat
                                                     : dma_sram_act_wr_data;   // zero-extend

            (* keep *) scratchpad_1024x128_freepdk45 u_act_sram (
                .clk0(clk),
                .csb0(~(ctrl_sram_act_rd_en || act_wr_en[si])),
                .web0(~act_wr_en[si]),
                .addr0(act_addr[si]),
                .din0(act_wr_data[si]),
                .dout0(act_rd_data[si])
            );
        end

        // ── Accumulator SRAM bank (16 x 16 KB = 256 KB = 2 groups x 8, ping-pong) ──
        //   addr[12:10] selects a scratchpad within the group (8 = one 1024-bit word)
        //   addr[13]    selects the group (0 = A / 1 = B) so DMA can read A while FP16/VecUnit writes B
        for (si = 0; si < ACC_SRAM_COUNT; si = si + 1) begin : acc_sram_gen
            wire acc_is_grpB = (si >= ACC_GROUP);                    // si 8..15 belong to group B
            wire [2:0] acc_lane = si[2:0];                            // index within the group

            assign acc_wr_en[si] = ctrl_sram_acc_wr_en
                                 && (acc_combined_addr[12:10] == acc_lane)
                                 && (acc_combined_addr[13]    == acc_is_grpB);
            assign acc_addr[si]  = acc_combined_addr[9:0];

            // Write data: FP16 array output (FP16_COMPUTE) or VecUnit output
            //   both are 1024 bit = 8 x 128 bit, sliced by the within-group index
            assign acc_wr_data[si] = fp16_running ? fp16_psums_out[acc_lane*128 +: 128]
                                                  : vec_data_out_flat[acc_lane*`SRAM_WIDTH +: `SRAM_WIDTH];

            // csb: during a read all 8 of this group are enabled (wide word); writes select one
            wire acc_rd_grp = (ctrl_sram_acc_rd_en || dma_sram_acc_rd_en)
                           && (acc_combined_addr[13] == acc_is_grpB);
            scratchpad_1024x128_freepdk45 u_acc_sram (
                .clk0(clk),
                .csb0(~(acc_rd_grp || acc_wr_en[si])),
                .web0(~acc_wr_en[si]),
                .addr0(acc_addr[si]),
                .din0(acc_wr_data[si]),
                .dout0(acc_rd_data[si])
            );
        end
    endgenerate

    // ── DMA accumulator read data (to TL response path) ──
    //   STORE phase: select one scratchpad from the current read group
    wire [3:0] dma_acc_rd_scratchpad = {acc_combined_addr[13], acc_combined_addr[12:10]};
    wire [127:0] dma_acc_rd_raw;
    assign dma_acc_rd_raw = acc_rd_flat[dma_acc_rd_scratchpad*`SRAM_WIDTH +: `SRAM_WIDTH];
    // Pad 128-bit SRAM word to 64-bit TL data width (DMA reads lower 64 bits per access)
    // DMA reads lower 64 bits of 128-bit SRAM word (SRAM_WIDTH > TL_DATA_WIDTH)
assign dma_sram_acc_rd_data = dma_acc_rd_raw[`TL_DATA_WIDTH-1:0];

    // ═══════════════════════════════════════════════════════════════
    // Compute Pipeline Datapath
    // ═══════════════════════════════════════════════════════════════

    // ── MsPE Array: activations from SRAM, weights from SRAM ──
    // Activations: 32 rows × 8 bits = 256 bits. From act SRAM.
    // Weights: 32×32 grid × 8 bits = 8192 bits, loaded in 8 × 1024-bit reads.
    wire [31:0][7:0]       mspe_activations;
    wire [31:0][31:0]      mspe_partial_sums;   // 32 columns

    // Assemble MsPE activations (32 rows × 8 bits = 256 bits) from act SRAM bank
    // 8 scratchpads × 128 bits = 1024 bits/cycle; use lower 2 scratchpads for 256 bits
    assign mspe_activations = {act_rd_data[1], act_rd_data[0]};

    mspe_array_32x32 u_mspe_array (
        .clk          (clk),
        .rst_n        (rst_n),
        .mode         (mspe_mode[0]),
        .start        (mspe_start),
        .accumulate   (1'b1),
        // weight load: the port is only 1024 bit; buffer and load FSM live inside the array
        .wt_load_req  (mspe_wt_load_req),
        .wt_wide      (mspe_wt_wide),
        .wt_load_busy (mspe_wt_load_busy),
        .wt_load_idx  (mspe_wt_load_idx),
        .activations  (mspe_activations),
        .partial_sums (mspe_partial_sums)
    );

    // ═══════════════════════════════════════════════════════════════
    // MsPE weight wide read: assemble the 8 scratchpad douts into 1024 bit for the array
    // ═══════════════════════════════════════════════════════════════
    //   The buffer and load FSM are inside mspe_array_32x32; the top level only:
    //     1) forms the SRAM row address from wt_load_idx (see wt_combined_addr)
    //     2) assembles the 8 dout values into the wide word
    //   Benefit: MsPE block IO pins 9,477 -> 2,314; top-level cross-macro wires down 87.5%.
    assign mspe_wt_wide = {wt_rd_data[7], wt_rd_data[6], wt_rd_data[5], wt_rd_data[4],
                           wt_rd_data[3], wt_rd_data[2], wt_rd_data[1], wt_rd_data[0]};

    // ── FP16 PE Array (32×32 dense FP16 matmul) ──
    wire [32*16-1:0]  fp16_activations;
    wire [32*16-1:0]  fp16_weights;
    wire [32*32-1:0]  fp16_psums_in = 0;

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

    // ── QuanRedPipe: MsPE INT32 partial sums (32 channels) → FP32 dequantized ──
    wire [31:0][31:0] quan_red_inputs;
    wire [31:0][31:0] quan_red_results;
    wire              quan_red_valid_out;

    // Direct MsPE → QuanRedPipe connection (no INT8 mux)
    assign quan_red_inputs = mspe_partial_sums;

    // ── Quantify scale latch: capture per-channel scales for QuanRedPipe ──
    reg [31:0][7:0] quant_scales_latched;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            quant_scales_latched <= 32'd0;
        else if (quant_valid_out)
            quant_scales_latched <= quant_scales_out;
    end

    // Scale mux: use latched quantify scales when available, else CSR
    wire [31:0] qrp_w_scale;
    assign qrp_w_scale = quant_scales_latched[0] != 8'd0 ?
                         {24'd0, quant_scales_latched[0]} :  // FP32 from Q8 scale
                         csr_w_scale;

    quan_red_pipe #(
        .N_CHAN(32)
    ) u_quan_red_pipe (
        .clk             (clk),
        .rst_n           (rst_n),
        .partial_sums_in (quan_red_inputs),
        .valid_in        (quan_red_pipe_start),
        .w_scale         (qrp_w_scale),
        .w_bias          (csr_w_bias),
        .w_sup_scale     (csr_w_sup_scale),
        .w_sup_bias      (csr_w_sup_bias),
        .channel_idx     (8'd0),
        .results_out     (quan_red_results),
        .valid_out       (quan_red_valid_out)
    );

    // ── Vector Register Stage ──
    // Captures all 16 QuanRedPipe channels, feeds VectorUnit directly.
    // 16 channels fit in one 32-lane batch (no dual-batch needed).
    vector_reg u_vector_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (quan_red_results),
        .valid_in  (quan_red_valid_out),
        .enable    (vector_unit_start),
        .ready_in  (vec_result_ready),
        .data_out  (vec_reg_data_out),
        .batch_sel (vec_reg_batch_sel),
        .valid_out (vec_reg_valid_out),
        .done      (vec_reg_done)
    );

    // ── Vector Unit (32-lane FP32 SIMD) ──
    // Takes QuanRedPipe outputs for element-wise post-processing
    wire [31:0][31:0] vec_data_in;
    wire              vec_result_valid;
    wire              vec_done;
    wire              vec_busy;

    // ── Vector Register Stage: captures all 16 QuanRedPipe outputs ──

    // 16 QuanRedPipe results → 32-lane vector unit (single batch, upper lanes zero)
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
    wire [15:0][15:0] quant_fp16_in;
    wire              quant_ready_out;

    // Input from activation SRAM: 16 channels × 16-bit FP16 = 256 bits
    // Assemble from 2 activation SRAM words (128 bits each)
    wire [255:0] quant_fp16_flat;
    assign quant_fp16_flat = {act_rd_data[1], act_rd_data[0]};
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
        .mspe_wt_load_req     (mspe_wt_load_req),     // weight-load handshake
        .mspe_wt_base_addr    (mspe_wt_base_addr),
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
    localparam MSPE_DRAIN_CYCLES  = 32 + 32;   // 64 cycles for 32x32 MsPE
    localparam FP16_DRAIN_CYCLES  = 32 + 32;   // 64 cycles for 32x32 FP16

    reg [7:0]  mspe_cycle_cnt;
    reg [7:0]  fp16_cycle_cnt;
    reg        mspe_running;

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

    // ── Master DMA: DSA-initiated DRAM access ──
    tl_master_dma #(.DW(64)) u_dma_master (
        .clk(clk), .rst_n(rst_n),
        .start(m_dma_start), .rw(m_dma_rw),
        .base_addr(m_dma_base_addr), .n_words(m_dma_n_words),
        .busy(m_dma_busy), .done(m_dma_done), .checksum(m_dma_checksum), .last_data(),
        .a_valid(m_a_valid), .a_ready(m_a_ready), .a_opcode(m_a_opcode), .a_param(),
        .a_size(m_a_size), .a_source(m_a_source), .a_address(m_a_address),
        .a_mask(), .a_data(m_a_data),
        .d_valid(m_d_valid), .d_ready(m_d_ready), .d_opcode(m_d_opcode),
        .d_param(), .d_size(), .d_source(m_d_source), .d_data(m_d_data),
        .d_denied(1'b0), .d_corrupt(1'b0)
    );

endmodule
