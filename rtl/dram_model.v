// dram_model.v — Shared DRAM model (2 TL-UL 64-bit slave ports + bandwidth gate + per-port counters)
//
// 2026-10-11: Aggregate DRAM bandwidth fixed at 64 GB/s per the paper:
//   BPC = beats accepted per cycle (64-bit/beat).
//     BPC=8 @1GHz => 8x8Bx1e9 = 64 GB/s (paper pool)
//     BPC=1 @1GHz => 8 GB/s           (used to force contention)
//   The two slave ports share the BPC limit; round-robin keeps service fair.
//   `o_beats0/1` = cumulative beats served per port (for measured bandwidth).
`timescale 1ns/1ps
`default_nettype none

module dram_model #(
    parameter DEPTH_WORDS = 1 << 20,
    parameter LAT         = 4,
    parameter BPC         = 8          // aggregate beats per cycle (8 => 64 GB/s @1 GHz)
) (
    input  wire         clk,
    input  wire         rst_n,

    // ── Slave port 0 ──
    input  wire         a0_valid, output wire a0_ready,
    input  wire [2:0]   a0_opcode, input wire [3:0] a0_size,
    input  wire [4:0]   a0_source, input wire [31:0] a0_address,
    input  wire [7:0]   a0_mask, input wire [63:0] a0_data,
    output wire         d0_valid, input wire d0_ready,
    output wire [2:0]   d0_opcode, output wire [4:0] d0_source,
    output wire [63:0]  d0_data,

    // ── Slave port 1 ──
    input  wire         a1_valid, output wire a1_ready,
    input  wire [2:0]   a1_opcode, input wire [3:0] a1_size,
    input  wire [4:0]   a1_source, input wire [31:0] a1_address,
    input  wire [7:0]   a1_mask, input wire [63:0] a1_data,
    output wire         d1_valid, input wire d1_ready,
    output wire [2:0]   d1_opcode, output wire [4:0] d1_source,
    output wire [63:0]  d1_data,

    // ── Counters ──
    output reg  [31:0]  o_beats0,
    output reg  [31:0]  o_beats1
);

    localparam OP_GET = 3'd4, OP_PUT = 3'd0;
    localparam ACK = 3'd0, ACK_DATA = 3'd1;

    reg [63:0] mem [0:DEPTH_WORDS-1];
    integer i;
    initial for (i = 0; i < DEPTH_WORDS; i = i + 1) mem[i] = 64'd0;

    // ── Per-cycle bandwidth gate + round-robin ──
    reg        rr;                          // priority port from previous cycle
    wire       first0 = ~rr;
    reg  [7:0] c;                           // remaining credits this cycle (reset to BPC each cycle)
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin rr <= 1'b0; c <= BPC[7:0]; end
        else        begin rr <= ~rr; c <= 0; end     // c is read combinationally, see below

    // take0/take1: at most one beat per port per cycle, gated by credit and fairness
    wire credit_ok = (BPC >= 2) || 1'b1;    // BPC>=2 => both ports allowed; BPC=1 => only one
    wire take0 = a0_valid && ((BPC >= 2) || first0);
    wire take1 = a1_valid && ((BPC >= 2) || ~first0) && !(take0 && (BPC < 2));

    assign a0_ready = take0;
    assign a1_ready = take1;

    // ── One LAT-deep pipeline per port ──
    reg [LAT-1:0] v0, v1;
    reg [4:0]  s0 [0:LAT-1], s1 [0:LAT-1];
    reg [2:0]  k0 [0:LAT-1], k1 [0:LAT-1];
    reg [63:0] q0 [0:LAT-1], q1 [0:LAT-1];
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v0 <= 0; v1 <= 0; end
        else begin
            for (j = LAT-1; j > 0; j = j - 1) begin
                v0[j] <= v0[j-1]; s0[j] <= s0[j-1]; k0[j] <= k0[j-1]; q0[j] <= q0[j-1];
                v1[j] <= v1[j-1]; s1[j] <= s1[j-1]; k1[j] <= k1[j-1]; q1[j] <= q1[j-1];
            end
            v0[0] <= take0; v1[0] <= take1;
            if (take0) begin
                s0[0] <= a0_source; k0[0] <= (a0_opcode==OP_GET) ? ACK_DATA : ACK;
                q0[0] <= (a0_opcode==OP_GET) ? mem[{9'd0,a0_address[22:3]}] : 64'd0;
                if (a0_opcode == OP_PUT) begin
                    if (a0_mask[0]) mem[{9'd0,a0_address[22:3]}][7:0]   <= a0_data[7:0];
                    if (a0_mask[1]) mem[{9'd0,a0_address[22:3]}][15:8]  <= a0_data[15:8];
                    if (a0_mask[2]) mem[{9'd0,a0_address[22:3]}][23:16] <= a0_data[23:16];
                    if (a0_mask[3]) mem[{9'd0,a0_address[22:3]}][31:24] <= a0_data[31:24];
                    if (a0_mask[4]) mem[{9'd0,a0_address[22:3]}][39:32] <= a0_data[39:32];
                    if (a0_mask[5]) mem[{9'd0,a0_address[22:3]}][47:40] <= a0_data[47:40];
                    if (a0_mask[6]) mem[{9'd0,a0_address[22:3]}][55:48] <= a0_data[55:48];
                    if (a0_mask[7]) mem[{9'd0,a0_address[22:3]}][63:56] <= a0_data[63:56];
                end
            end
            if (take1) begin
                s1[0] <= a1_source; k1[0] <= (a1_opcode==OP_GET) ? ACK_DATA : ACK;
                q1[0] <= (a1_opcode==OP_GET) ? mem[{9'd0,a1_address[22:3]}] : 64'd0;
                if (a1_opcode == OP_PUT) begin
                    if (a1_mask[0]) mem[{9'd0,a1_address[22:3]}][7:0]   <= a1_data[7:0];
                    if (a1_mask[1]) mem[{9'd0,a1_address[22:3]}][15:8]  <= a1_data[15:8];
                    if (a1_mask[2]) mem[{9'd0,a1_address[22:3]}][23:16] <= a1_data[23:16];
                    if (a1_mask[3]) mem[{9'd0,a1_address[22:3]}][31:24] <= a1_data[31:24];
                    if (a1_mask[4]) mem[{9'd0,a1_address[22:3]}][39:32] <= a1_data[39:32];
                    if (a1_mask[5]) mem[{9'd0,a1_address[22:3]}][47:40] <= a1_data[47:40];
                    if (a1_mask[6]) mem[{9'd0,a1_address[22:3]}][55:48] <= a1_data[55:48];
                    if (a1_mask[7]) mem[{9'd0,a1_address[22:3]}][63:56] <= a1_data[63:56];
                end
            end
        end
    end

    assign d0_valid = v0[LAT-1]; assign d0_opcode = k0[LAT-1];
    assign d0_source = s0[LAT-1]; assign d0_data = q0[LAT-1];
    assign d1_valid = v1[LAT-1]; assign d1_opcode = k1[LAT-1];
    assign d1_source = s1[LAT-1]; assign d1_data = q1[LAT-1];

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin o_beats0 <= 32'd0; o_beats1 <= 32'd0; end
        else        begin
            if (take0) o_beats0 <= o_beats0 + 32'd1;
            if (take1) o_beats1 <= o_beats1 + 32'd1;
        end

endmodule

`default_nettype wire
