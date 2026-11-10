// tl_master_dma.v — Multi-outstanding TileLink-UL master DMA
//
// 2026-10-10: Changed from single-outstanding to up to DEPTH requests in flight
//   (a_source = issued round-robin).
//   The master no longer waits for each D; it can sustain one issue per cycle,
//   exposing the direct-DRAM bandwidth limit.
//   Reads are XOR-aggregated (order-independent), so no per-source reassembly.
`timescale 1ns/1ps
`default_nettype none

module tl_master_dma #(
    parameter DW    = 64,
    parameter DEPTH = 16        // in-flight depth (>= mem LAT avoids throttling)
) (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         start,
    input  wire         rw,          // 0 = read from DRAM, 1 = write to DRAM
    input  wire [31:0]  base_addr,
    input  wire [15:0]  n_words,
    output reg          busy,
    output reg          done,
    output reg  [63:0]  checksum,
    output reg  [63:0]  last_data,

    output wire         a_valid,
    input  wire         a_ready,
    output reg  [2:0]   a_opcode,
    output wire [2:0]   a_param,
    output reg  [3:0]   a_size,
    output reg  [4:0]   a_source,
    output reg  [31:0]  a_address,
    output wire [DW/8-1:0] a_mask,
    output reg  [DW-1:0]   a_data,

    input  wire         d_valid,
    output wire         d_ready,
    input  wire [2:0]   d_opcode,
    input  wire [1:0]   d_param,
    input  wire [3:0]   d_size,
    input  wire [4:0]   d_source,
    input  wire [DW-1:0] d_data,
    input  wire         d_denied,
    input  wire         d_corrupt
);

    localparam OP_GET = 3'd4, OP_PUT = 3'd0;
    localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;

    reg [1:0]  st;
    reg [15:0] n_q;
    reg [31:0] base_q;
    reg        rw_q;
    reg [15:0] issued, outp;

    assign a_param = 3'd0;
    assign a_mask  = {{(DW/8-8){1'b0}}, 8'hFF};
    assign d_ready = 1'b1;

    // Issue while payload remains and in-flight count < DEPTH
    assign a_valid = (st == S_RUN) && (issued < n_q) && (outp < DEPTH);

    wire issue_ok = a_valid & a_ready;
    wire d_ok     = d_valid & d_ready;

    always @(*) begin
        a_opcode  = rw_q ? OP_PUT : OP_GET;
        a_size    = 4'd3;
        a_source  = issued[4:0];
        a_address = base_q + (issued * 32'd8);
        a_data    = {{(DW-8){1'b0}}, issued[7:0]};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            issued <= 16'd0; outp <= 16'd0; checksum <= 64'd0; last_data <= 64'd0;
            n_q <= 16'd0; base_q <= 32'd0; rw_q <= 1'b0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1; issued <= 16'd0; outp <= 16'd0; checksum <= 64'd0;
                        n_q <= n_words; base_q <= base_addr; rw_q <= rw;
                        st <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (issue_ok) issued <= issued + 16'd1;
                    if (d_ok) begin
                        checksum <= checksum ^ d_data;
                        last_data <= d_data;
                    end
                    // In-flight count: unchanged when issue and D complete in the same cycle
                    if (issue_ok && !d_ok)      outp <= outp + 16'd1;
                    else if (!issue_ok && d_ok) outp <= outp - 16'd1;
                    if (issued >= n_q && outp == 16'd0) st <= S_DONE;
                end
                S_DONE: begin
                    done <= 1'b1; busy <= 1'b0;
                    if (!start) st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
