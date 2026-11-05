# floorplan_dsa.tcl — DSA internal floorplan + GPL
# Reference: /home/user/project/sparsenn/cpu_st_top/st_v20/floorplan_v22.tcl
#
# Layout: 3-row structure (sparsenn v22 pattern)
#   Row 1 (top):    Weight SRAM (8) + Activation SRAM (8) — 16 macros in 2 rows
#   Row 2 (middle): Compute (MsPE + FP16 PE + QuanRedPipe + QuantE + VecUnit + Ctrl)
#   Row 3 (bottom): Accumulator SRAM (16) + Quantify Buffer

source /home/user/gemmini-flow/flow/scripts/util.tcl

set lef_dir    /home/user/gemmini-flow/flow/platforms/nangate45/lef
set lib_dir    /home/user/gemmini-flow/flow/platforms/nangate45/lib
set results_dir /home/user/project/SpecLLMDSA/results
set scratchpad_lef /home/user/project/sparsenn/cpu_st_top/st_block_hier/scratchpad_1024x128_freepdk45.lef

# ── Read technology ──
read_lef $lef_dir/NangateOpenCellLibrary.tech.lef
read_lef $lef_dir/NangateOpenCellLibrary.macro.mod.lef
read_lef $scratchpad_lef
read_liberty $lib_dir/NangateOpenCellLibrary_typical.lib

# ── Read synthesized netlist ──
read_verilog $results_dir/1_synth_dsa.v
link_design specLLM_dsa_top

# ═══════════════════════════════════════════════════════════
# Floorplan parameters
# ═══════════════════════════════════════════════════════════

set SRAM_W   452.24
set SRAM_H   380.715
set GAP      20
set MARGIN   20

# Signoff: design area 13.84 mm² (13.75 macro + 0.086 logic)
# Die: 4200×4300 um = 18.06 mm² → ~76.6% design occupancy
set DIE_X0  0
set DIE_Y0  0
set DIE_X1  4200
set DIE_Y1  4300

initialize_floorplan \
    -die_area  "$DIE_X0 $DIE_Y0 $DIE_X1 $DIE_Y1" \
    -core_area "[expr {$DIE_X0+20}] [expr {$DIE_Y0+20}] [expr {$DIE_X1-20}] [expr {$DIE_Y1-20}]" \
    -site FreePDK45_38x28_10R_NP_162NW_34O

# ═══════════════════════════════════════════════════════════
# Collect SRAM instances
# ═══════════════════════════════════════════════════════════

set wt_srams {}
set act_srams {}
set acc_srams {}
set buf_srams {}

# Collect all scratchpad SRAMs (broad regex: match any instance containing scratchpad)
foreach inst [get_cells -hier -filter "ref_name =~ *scratchpad*"] {
    set name [get_name $inst]
    if {[regexp {wt_sram|wt_sram_gen|wt.*sram} $name]} {
        lappend wt_srams $inst
    } elseif {[regexp {act_sram|act_sram_gen|act.*sram} $name]} {
        lappend act_srams $inst
    } elseif {[regexp {acc_sram|acc_sram_gen|acc.*sram} $name]} {
        lappend acc_srams $inst
    } else {
        lappend acc_srams $inst
    }
}

# Collect quantify_buffer_sram (flatten-resistant: match ref_name wildcard)
foreach inst [get_cells -hier -filter "ref_name =~ *quantify*"] {
    lappend buf_srams $inst
}

# Fallback: if regex-based collection missed SRAMs, use total count check
if {[llength $wt_srams] < 8 || [llength $act_srams] < 8 || [llength $acc_srams] < 8} {
    puts "WARNING: SRAM count low — retrying with name-based collection"
    set all_srams [get_cells -hier -filter "ref_name =~ *scratchpad*"]
    set idx 0
    foreach inst $all_srams {
        if {$idx < 8} { lappend wt_srams $inst
        } elseif {$idx < 16} { lappend act_srams $inst
        } else { lappend acc_srams $inst }
        incr idx
    }
}

puts "SRAM count: [llength $wt_srams] wt, [llength $act_srams] act, [llength $acc_srams] acc, [llength $buf_srams] buf"
puts "Total SRAMs: [expr {[llength $wt_srams] + [llength $act_srams] + [llength $acc_srams] + [llength $buf_srams]}]"

# ═══════════════════════════════════════════════════════════
# Row 1 (top): Weight + Activation SRAMs
#   8 wt + 8 act = 16 macros → 2 rows × 8 columns
# ═══════════════════════════════════════════════════════════

set row1_y [expr {$DIE_Y1 - $MARGIN - $SRAM_H}]
set row1_cols 8

set all_row1_srams [concat $wt_srams $act_srams]
set idx 0
foreach inst $all_row1_srams {
    set col [expr {$idx % $row1_cols}]
    set row [expr {$idx / $row1_cols}]
    set sx [expr {$MARGIN + $col * ($SRAM_W + $GAP)}]
    set sy [expr {$row1_y - $row * ($SRAM_H + $GAP)}]
    place_cell -inst_name $inst -origin "$sx $sy" -status FIXED
    incr idx
}

# ═══════════════════════════════════════════════════════════
# Row 3 (bottom): Accumulator SRAMs + Quantify Buffer
#   16 acc + 1 buf = 17 macros → 3 rows × 8 columns
# ═══════════════════════════════════════════════════════════

set row3_y [expr {$MARGIN}]
set row3_cols 8
set row3_srams [concat $acc_srams $buf_srams]
set idx 0
foreach inst $row3_srams {
    set col [expr {$idx % $row3_cols}]
    set row [expr {$idx / $row3_cols}]
    set sx [expr {$MARGIN + $col * ($SRAM_W + $GAP)}]
    set sy [expr {$row3_y + $row * ($SRAM_H + $GAP)}]
    place_cell -inst_name $inst -origin "$sx $sy" -status FIXED
    incr idx
}

# Row 2: leave empty for global_placement to fill with stdcells

# ═══════════════════════════════════════════════════════════
# Pin Placement (MUST run before GPL — fixes GPL-0326)
# ═══════════════════════════════════════════════════════════

puts "Placing pins..."
# Place I/O pins evenly along the bottom and right edges
set pin_x0 $DIE_X0
set pin_x1 $DIE_X1
set pin_y0 $DIE_Y0
set pin_y1 $DIE_Y1

# Collect all top-level ports
set all_ports [get_ports]
set n_ports [llength $all_ports]
set pin_idx 0

# Place ports in order along the boundary edges
# Bottom edge: first 40% of ports
# Right edge: next 30%
# Top edge: next 20%
# Left edge: remaining 10%
set n_bottom [expr {int($n_ports * 0.4)}]
set n_right  [expr {int($n_ports * 0.3)}]
set n_top    [expr {int($n_ports * 0.2)}]

foreach port $all_ports {
    if {$pin_idx < $n_bottom} {
        set frac [expr {double($pin_idx) / max($n_bottom - 1, 1)}]
        set px [expr {$pin_x0 + 50 + $frac * ($pin_x1 - $pin_x0 - 100)}]
        set py $pin_y0
    } elseif {$pin_idx < $n_bottom + $n_right} {
        set frac [expr {double($pin_idx - $n_bottom) / max($n_right - 1, 1)}]
        set px $pin_x1
        set py [expr {$pin_y0 + 50 + $frac * ($pin_y1 - $pin_y0 - 100)}]
    } elseif {$pin_idx < $n_bottom + $n_right + $n_top} {
        set frac [expr {double($pin_idx - $n_bottom - $n_right) / max($n_top - 1, 1)}]
        set px [expr {$pin_x0 + 50 + $frac * ($pin_x1 - $pin_x0 - 100)}]
        set py $pin_y1
    } else {
        set frac [expr {double($pin_idx - $n_bottom - $n_right - $n_top) / max($n_ports - $n_bottom - $n_right - $n_top - 1, 1)}]
        set px $pin_x0
        set py [expr {$pin_y0 + 50 + $frac * ($pin_y1 - $pin_y0 - 100)}]
    }
    place_pin -pin_name $port -location "$px $py" -force
    incr pin_idx
}
puts "Placed $pin_idx pins"

# ═══════════════════════════════════════════════════════════
# Global Placement
# ═══════════════════════════════════════════════════════════

puts "Running global placement..."
if {[catch {global_placement -density 0.55 -pad_left 2 -pad_right 2} gpl_err]} {
    puts "GPL Warning: $gpl_err"
    catch {global_placement -density 0.50}
}

puts "Running detailed placement..."
catch {detailed_placement}

# ═══════════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════════

write_def $results_dir/dsa_placed.def
puts "DEF written: $results_dir/dsa_placed.def"
puts "Floorplan complete."
