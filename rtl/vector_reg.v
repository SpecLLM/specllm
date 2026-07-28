// vector_reg.v — 64-Channel Register Stage for QuanRedPipe → VectorUnit
//
// Captures ALL 64 QuanRedPipe output channels in a register bank, then
// feeds them to the 32-lane VectorUnit in two batches (lower 32, upper 32).
// This ensures the full 64-column MsPE and 32-column INT8 arrays are
// consumed, preventing ABC from optimizing away "dead" upper columns.
//
// FSM: IDLE → CAPTURE → SEND_LOW → SEND_HIGH → DONE → IDLE

`include "dsa_params.vh"

module vector_reg (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── Input from QuanRedPipe ──
    input  wire [63:0][31:0]       data_in,
    input  wire                     valid_in,

    // ── Enable: gates advancement past CAPTURE ──
    input  wire                     enable,

    // ── Output to VectorUnit (32-lane batches) ──
    output reg  [31:0][31:0]       data_out,
    output reg                      batch_sel,     // 0 = lower 32, 1 = upper 32
    output reg                      valid_out,
    output reg                      done
);

    // ── FSM state encoding ──
    localparam S_IDLE      = 3'd0;
    localparam S_CAPTURE   = 3'd1;
    localparam S_SEND_LOW  = 3'd2;
    localparam S_SEND_HIGH = 3'd3;
    localparam S_DONE      = 3'd4;

    reg [2:0] state, next_state;

    // ── Internal register bank: 64 channels × 32 bits ──
    reg [63:0][31:0] reg_bank;

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
                if (valid_in)
                    next_state = S_CAPTURE;

            S_CAPTURE:
                if (enable)
                    next_state = S_SEND_LOW;

            S_SEND_LOW:
                next_state = S_SEND_HIGH;

            S_SEND_HIGH:
                next_state = S_DONE;

            S_DONE:
                next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

    // ── Output logic ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_bank  <= '{default: 32'd0};
            data_out  <= '{default: 32'd0};
            batch_sel <= 1'b0;
            valid_out <= 1'b0;
            done      <= 1'b0;
        end else begin
            // Default deassertions (pulsed signals)
            valid_out <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    batch_sel <= 1'b0;
                end

                S_CAPTURE: begin
                    // Latch all 64 channels into register bank
                    reg_bank  <= data_in;
                    batch_sel <= 1'b0;
                end

                S_SEND_LOW: begin
                    // Output lower 32 channels (channels [31:0])
                    data_out  <= reg_bank[31:0];
                    batch_sel <= 1'b0;
                    valid_out <= 1'b1;
                end

                S_SEND_HIGH: begin
                    // Output upper 32 channels (channels [63:32])
                    data_out  <= reg_bank[63:32];
                    batch_sel <= 1'b1;
                    valid_out <= 1'b1;
                end

                S_DONE: begin
                    done <= 1'b1;
                end

                default: ;
            endcase
        end
    end

endmodule
