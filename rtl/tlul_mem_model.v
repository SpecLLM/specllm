// tlul_mem_model.v — Pipelined TileLink-UL 64-bit DRAM model
//
// 2026-10-10: Changed from single-beat to a LAT-deep pipeline: accepts one A
//   beat per cycle and returns D after LAT cycles.
//   Steady-state throughput = 1 beat/cycle (= 8 B/cyc), the direct-DRAM limit.
//   (The old S_IDLE->CNT->RESP model took ~6 cycles per beat and hid master MLP gains.)
`timescale 1ns/1ps
`default_nettype none

module tlul_mem_model #(
    parameter DEPTH_WORDS = 1 << 20,   // default 1M x 8B = 8 MB
    parameter LAT         = 4          // A->D latency (pipeline depth)
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         a_valid,
    output wire         a_ready,
    input  wire [2:0]   a_opcode,
    input  wire [2:0]   a_param,
    input  wire [3:0]   a_size,
    input  wire [4:0]   a_source,
    input  wire [31:0]  a_address,
    input  wire [7:0]   a_mask,
    input  wire [63:0]  a_data,
    output wire         d_valid,
    input  wire         d_ready,
    output wire [2:0]   d_opcode,
    output wire [1:0]   d_param,
    output wire [3:0]   d_size,
    output wire [4:0]   d_source,
    output wire [63:0]  d_data,
    output wire         d_denied,
    output wire         d_corrupt
);

    localparam OP_GET = 3'd4, OP_PUT = 3'd0;
    localparam ACK = 3'd0, ACK_DATA = 3'd1;

    reg [63:0] mem [0:DEPTH_WORDS-1];
    integer i;
    initial for (i = 0; i < DEPTH_WORDS; i = i + 1) mem[i] = 64'd0;

    wire [31:0] widx = {9'd0, a_address[22:3]};
    wire        is_get = (a_opcode == OP_GET);
    wire        is_put = (a_opcode == OP_PUT);

    assign a_ready = 1'b1;              // accept one beat per cycle

    reg [LAT-1:0] dv;
    reg [4:0]     dsrc [0:LAT-1];
    reg [2:0]     dack [0:LAT-1];
    reg [63:0]    ddat [0:LAT-1];
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv <= {LAT{1'b0}};
        end else begin
            for (k = LAT-1; k > 0; k = k - 1) begin
                dv[k]   <= dv[k-1];
                dsrc[k] <= dsrc[k-1];
                dack[k] <= dack[k-1];
                ddat[k] <= ddat[k-1];
            end
            dv[0] <= a_valid & a_ready;
            if (a_valid && a_ready) begin
                dsrc[0] <= a_source;
                dack[0] <= is_get ? ACK_DATA : ACK;
                ddat[0] <= is_get ? mem[widx] : 64'd0;
                if (is_put) begin
                    if (a_mask[0]) mem[widx][7:0]   <= a_data[7:0];
                    if (a_mask[1]) mem[widx][15:8]  <= a_data[15:8];
                    if (a_mask[2]) mem[widx][23:16] <= a_data[23:16];
                    if (a_mask[3]) mem[widx][31:24] <= a_data[31:24];
                    if (a_mask[4]) mem[widx][39:32] <= a_data[39:32];
                    if (a_mask[5]) mem[widx][47:40] <= a_data[47:40];
                    if (a_mask[6]) mem[widx][55:48] <= a_data[55:48];
                    if (a_mask[7]) mem[widx][63:56] <= a_data[63:56];
                end
            end
        end
    end

    assign d_valid   = dv[LAT-1];
    assign d_opcode  = dack[LAT-1];
    assign d_param   = 2'd0;
    assign d_size    = 4'd3;
    assign d_source  = dsrc[LAT-1];
    assign d_data    = ddat[LAT-1];
    assign d_denied  = 1'b0;
    assign d_corrupt = 1'b0;

endmodule

`default_nettype wire
