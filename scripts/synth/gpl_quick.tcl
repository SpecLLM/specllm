# gpl_quick.tcl — Minimal GPL run for placement heatmap
# Goal: get stdcell coordinates into DEF, nothing else

set lef_dir    /home/user/gemmini-flow/flow/platforms/nangate45/lef
set lib_dir    /home/user/gemmini-flow/flow/platforms/nangate45/lib
set res_dir    /home/user/project/SpecLLMDSA/results
set sram_lef   /home/user/project/sparsenn/cpu_st_top/st_block_hier/scratchpad_1024x128_freepdk45.lef

# Read tech
read_lef $lef_dir/NangateOpenCellLibrary.tech.lef
read_lef $lef_dir/NangateOpenCellLibrary.macro.mod.lef
read_lef $sram_lef
read_liberty $lib_dir/NangateOpenCellLibrary_typical.lib

# Read netlist
read_verilog $res_dir/1_synth_dsa_flat.v
link_design specLLM_dsa_top

# Floorplan: 4200 x 4300 um (18.06 mm² die), 33 SRAM macros
initialize_floorplan \
    -die_area  "0 0 4200 4300" \
    -core_area "20 20 4180 4280" \
    -site FreePDK45_38x28_10R_NP_162NW_34O

# Place all SRAM macros at fixed positions
# Row 1 (top): 16 SRAMs (8 wt + 8 act) in 2 rows x 8 columns
# Row 3 (bottom): 16 acc + 1 buf in 3 rows x 8 columns
set SRAM_W 452.24
set SRAM_H 380.715
set GAP 20

set all_srams [get_cells -hier -filter "is_macro == 1 || ref_name =~ *scratchpad*"]
set sram_count [llength $all_srams]
puts "Found $sram_count SRAM macros"

set idx 0
foreach inst $all_srams {
    if {$idx < 16} {
        # Row 1 (top): 16 SRAMs, 2 rows x 8 cols
        set col [expr {$idx % 8}]
        set row [expr {$idx / 8}]
        set sx [expr {20 + $col * ($SRAM_W + $GAP)}]
        set sy [expr {4300 - 40 - $SRAM_H - $row * ($SRAM_H + $GAP)}]
    } else {
        # Row 3 (bottom): remaining SRAMs, 8 per row
        set n [expr {$idx - 16}]
        set col [expr {$n % 8}]
        set row [expr {$n / 8}]
        set sx [expr {20 + $col * ($SRAM_W + $GAP)}]
        set sy [expr {40 + $row * ($SRAM_H + $GAP)}]
    }
    set_dont_touch $inst true
    place_inst -inst_name $inst -location "$sx $sy" -status FIRM
    puts "  SRAM\[$idx\]: [get_name $inst] at ($sx, $sy)"
    incr idx
}

# Minimal GPL — just get cells placed
puts "Running global_placement..."
global_placement -density 0.50 -pad_left 2 -pad_right 2

puts "Running detailed_placement..."
detailed_placement

# Write DEF with placed coordinates
puts "Writing DEF..."
write_def $res_dir/dsa_placed_v2.def
puts "Done. DEF: $res_dir/dsa_placed_v2.def"
