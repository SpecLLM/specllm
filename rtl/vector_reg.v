// vector_reg.v — 32-Channel Register Stage for QuanRedPipe → VectorUnit
//
// Captures ALL 32 QuanRedPipe output channels in a register bank, then
// feeds them to the 32-lane VectorUnit in a single batch.
// 32 channels map to 32 vector lanes (full width).
//
// FSM: IDLE → CAPTURE → SEND → DONE → IDLE
// SEND state waits for ready_in (backpressure from VectorUnit) before
// asserting valid_out and advancing to DONE.

`include "dsa_params.vh"

module vector_reg (
    input  wire                     clk,
    input  wire                     rst_n,

    // ── Input from QuanRedPipe ──
    input  wire [31:0][31:0]       data_in,
    input  wire                     valid_in,

    // ── Enable: gates advancement past CAPTURE ──
    input  wire                     enable,

    // ── Flow control from downstream (VectorUnit data_ready) ──
    input  wire                     ready_in,

    // ── Output to VectorUnit (32-lane / 1024-b batch) ──
    output reg  [31:0][31:0]       data_out,
    output reg                      batch_sel,     // always 0 (single batch)
    output reg                      valid_out,
    output reg                      done
);

    // ── FSM state encoding ──
    localparam S_IDLE    = 3'd0;
    localparam S_CAPTURE = 3'd1;
    localparam S_SEND    = 3'd2;
    localparam S_DONE    = 3'd3;

    reg [2:0] state, next_state;

    // ── Internal register bank: 16 channels × 32 bits ──
    reg [31:0][31:0] reg_bank;

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
                    next_state = S_SEND;

            S_SEND:
                if (ready_in)
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
            reg_bank  <= {32{32'd0}};
            data_out  <= {32{32'd0}};
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
                    // Latch all 32 channels into register bank
                    reg_bank  <= data_in;
                    batch_sel <= 1'b0;
                end

                S_SEND: begin
                    if (ready_in) begin
                        // Output all 32 channels
                        data_out <= reg_bank;
                        batch_sel <= 1'b0;
                        valid_out <= 1'b1;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

                default: ;
            endcase
        end
    end

endmodule
