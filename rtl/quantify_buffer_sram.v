// quantify_buffer_sram.v — Behavioral SRAM for Quantify Engine buffer
// 128 depth × 1024 width (16 channels × 16-bit FP16)
//
// Instantiated at top level for hierarchical synthesis compatibility.
// Replaceable with real SRAM macro via blackbox surgery flow.

module quantify_buffer_sram (
    input  wire         clk,
    input  wire         wr_en,
    input  wire [6:0]   wr_addr,
    input  wire [1023:0] wr_data,
    input  wire [6:0]   rd_addr,
    output reg  [1023:0] rd_data
);

    parameter DEPTH = 128;
    parameter WIDTH = 1024;

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // Asynchronous read (for timing simplicity; register at consumer)
    always @(*) begin
        rd_data = mem[rd_addr];
    end

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

endmodule
