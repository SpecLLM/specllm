// tlul_arb4_ul: 4-to-1 TileLink round-robin arbiter, A/D channels only, depth 8 per master + source remapping
module tlul_arb4_ul (
  input  wire clk,
  input  wire rst_n,
  input  wire [3:0] m_a_valid,
  input  wire [4*32-1:0] m_a_bits_address,
  input  wire [3:0] m_a_bits_corrupt,
  input  wire [4*128-1:0] m_a_bits_data,
  input  wire [4*16-1:0] m_a_bits_mask,
  input  wire [4*3-1:0] m_a_bits_opcode,
  input  wire [4*3-1:0] m_a_bits_param,
  input  wire [4*4-1:0] m_a_bits_size,
  input  wire [4*5-1:0] m_a_bits_source,
  output wire [3:0] m_a_ready,
  output wire [3:0] m_d_valid,
  output wire [3:0] m_d_bits_corrupt,
  output wire [4*128-1:0] m_d_bits_data,
  output wire [3:0] m_d_bits_denied,
  output wire [4*3-1:0] m_d_bits_opcode,
  output wire [4*2-1:0] m_d_bits_param,
  output wire [4*4-1:0] m_d_bits_sink,
  output wire [4*4-1:0] m_d_bits_size,
  output wire [4*5-1:0] m_d_bits_source,
  input  wire [3:0] m_d_ready,
  output wire s_a_valid,
  output wire [31:0] s_a_bits_address,
  output wire s_a_bits_corrupt,
  output wire [127:0] s_a_bits_data,
  output wire [15:0] s_a_bits_mask,
  output wire [2:0] s_a_bits_opcode,
  output wire [2:0] s_a_bits_param,
  output wire [3:0] s_a_bits_size,
  output wire [4:0] s_a_bits_source,
  input  wire s_a_ready,
  input  wire s_d_valid,
  input  wire s_d_bits_corrupt,
  input  wire [127:0] s_d_bits_data,
  input  wire s_d_bits_denied,
  input  wire [2:0] s_d_bits_opcode,
  input  wire [1:0] s_d_bits_param,
  input  wire [3:0] s_d_bits_sink,
  input  wire [3:0] s_d_bits_size,
  input  wire [4:0] s_d_bits_source,
  output wire s_d_ready
);

  // ── N-deep: tag width = 5-bit source (3-bit seq per master → depth 8) + source remapping ──
  reg  [31:0] tag_busy;
  reg  [6:0]  tag_src [0:31];
  reg  [2:0]  seq [0:3];
  reg         gnt_v; reg [1:0] gnt_q, rr_q;
  wire [3:0]  req_ok;
  assign req_ok[0] = m_a_valid[0] & ~tag_busy[{2'd0, seq[0]}];
  assign req_ok[1] = m_a_valid[1] & ~tag_busy[{2'd1, seq[1]}];
  assign req_ok[2] = m_a_valid[2] & ~tag_busy[{2'd2, seq[2]}];
  assign req_ok[3] = m_a_valid[3] & ~tag_busy[{2'd3, seq[3]}];
  reg [1:0] pick;
  always @* begin
    if      (req_ok[rr_q])                 pick = rr_q;
    else if (req_ok[(rr_q+2'd1)&2'd3])     pick = rr_q + 2'd1;
    else if (req_ok[(rr_q+2'd2)&2'd3])     pick = rr_q + 2'd2;
    else                                   pick = rr_q + 2'd3;
  end
  wire [4:0] gnt_tag = {gnt_q, seq[gnt_q]};
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tag_busy<=32'd0; gnt_v<=1'b0; gnt_q<=2'd0; rr_q<=2'd0;
      seq[0]<=3'd0; seq[1]<=3'd0; seq[2]<=3'd0; seq[3]<=3'd0;
    end else begin
      if (!gnt_v && (|req_ok)) begin gnt_v<=1'b1; gnt_q<=pick; rr_q<=pick+2'd1; end
      if (gnt_v && s_a_ready) begin
        tag_busy[gnt_tag] <= 1'b1;
        tag_src[gnt_tag]  <= {gnt_q, m_a_bits_source[gnt_q*5 +: 5]};
        seq[gnt_q] <= seq[gnt_q] + 3'd1;
        gnt_v <= 1'b0;
      end
      if (s_d_valid && tag_busy[s_d_bits_source]) tag_busy[s_d_bits_source] <= 1'b0;
    end
  end
  assign s_a_valid = gnt_v;
  assign m_a_ready = {4{gnt_v & s_a_ready}} & (4'd1 << gnt_q);
  assign s_a_bits_address = m_a_bits_address[gnt_q*32 +: 32];
  assign s_a_bits_corrupt = m_a_bits_corrupt[gnt_q*1 +: 1];
  assign s_a_bits_data = m_a_bits_data[gnt_q*128 +: 128];
  assign s_a_bits_mask = m_a_bits_mask[gnt_q*16 +: 16];
  assign s_a_bits_opcode = m_a_bits_opcode[gnt_q*3 +: 3];
  assign s_a_bits_param = m_a_bits_param[gnt_q*3 +: 3];
  assign s_a_bits_size = m_a_bits_size[gnt_q*4 +: 4];
  assign s_a_bits_source = gnt_tag;
  wire dhit = tag_busy[s_d_bits_source];
  wire [1:0] dhit_m = tag_src[s_d_bits_source][6:5];
  assign s_d_ready = dhit ? m_d_ready[dhit_m] : 1'b1;
  assign m_d_valid = dhit ? (4'd1 << dhit_m) : 4'd0;
  assign m_d_bits_corrupt = {4{s_d_bits_corrupt}};
  assign m_d_bits_data = {4{s_d_bits_data}};
  assign m_d_bits_denied = {4{s_d_bits_denied}};
  assign m_d_bits_opcode = {4{s_d_bits_opcode}};
  assign m_d_bits_param = {4{s_d_bits_param}};
  assign m_d_bits_sink = {4{s_d_bits_sink}};
  assign m_d_bits_size = {4{s_d_bits_size}};
  assign m_d_bits_source = {4{tag_src[s_d_bits_source][4:0]}};
endmodule
