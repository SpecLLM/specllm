# SpecLLM DSA SoC — Layout Lessons Learned

## 1. RTL Connectivity: Every Output Must Have a Consumer

**教训:** 综合工具（Yosys `opt_clean`, ABC）会移除无消费者的逻辑。对于 blackbox（如 SRAM macro），无消费者的实例同样会被移除。

**案例:**
- 4 个 activation SRAM 被 `opt_clean -purge` 移除：MsPE 只用 `act_rd_data[3:0]`，indices 4-7 无消费者
- Quantify Engine 输出 `quant_q8_out`/`quant_scales_out` 完全悬空 → 整个模块被优化
- INT8 PE array 输出 `int8_psums_out` 悬空 → 模块被优化
- Vector Unit 输出 `vec_data_out` 悬空 → 模块被优化 + FSM 死锁

**排查方法:**
1. 对每个模块输出，追踪其到顶层端口或 SRAM 的通路
2. 确认每个 `output` 信号至少驱动一个 `input`
3. 综合后检查 `stat` 报告中的实例数量

**修复原则:**
- 数据通路必须形成闭环：SRAM → Compute → SRAM → DMA → TL Bus
- 不要用 debug port 作假消费者——接入真实数据通路
- 所有 8 个 act SRAM 分摊到 MsPE/INT8/Quantify 三条读通路

---

## 2. Yosys SystemVerilog 兼容性

### 2.1 禁止：表达式上的 part-select

```verilog
// ❌ Yosys read_verilog -sv 不支持
assign result = (abs_val << N)[22:0];

// ✅ 用中间 wire
reg [54:0] abs_val_shifted;
abs_val_shifted = abs_val << N;
assign result = abs_val_shifted[22:0];
```

**影响文件:** `quan_red_pipe.v`。sparsenn 的 `fp16_mul.v`, `fp32_add.v` 也有同样问题。

### 2.2 `+:` part-select 需要 `-sv` 或 Slang 前端

```verilog
// ✅ read_verilog -sv 支持
assign slice = data[addr*W +: W];
// ✅ read_slang --ignore-timing 也支持
```

### 2.3 Slang 严格检查（ORFS 流程）

Slang 比 `read_verilog -sv` 更严格，会检查：
- **声明顺序:** `reg` 必须在 `begin` 块中所有可执行语句之前
- **模块引用:** 所有实例化模块必须在编译列表中
- **常量表达:** replication count 不能为负数

---

## 3. DFF 映射：dfflibmap 不可靠，需要后处理

**教训:** Yosys `dfflibmap -liberty <nangate>` 不能保证所有 `$_DFF_*` 都映射到目标库 cell。

**现象:** 7030 个 DFF 被映射为 `_adff`/`_adffe`/`_dffe` 而非 Nangate 的 `DFFR_X1`/`SDFFR_X1`/`SDFF_X1`。

**影响链:**
```
dfflibmap 失败 
  → _adff/_adffe cell (7030 个)
  → 需要 dff_stubs.lef (7030 个假 MACRO)
  → place_pins 遭遇 7030 个未放置 MACRO → PPL-0015 → PPL-0021
  → GPL-0326 → 全部 stdcell placement 被跳过
```

**修复（Python 后处理）:**
```python
content = re.sub(r'\b_adff\b\s+(\S+)\s*\(', r'DFFR_X1 \1 (', content)
content = re.sub(r'\b_adffe\b\s+(\S+)\s*\(', r'SDFFR_X1 \1 (', content)
content = re.sub(r'\b_dffe\b\s+(\S+)\s*\(', r'SDFF_X1 \1 (', content)
```

**映射表:**
| Yosys 输出 | Nangate 等价 | 功能 |
|-----------|-------------|------|
| `_adff` | `DFFR_X1` | DFF + async reset (active low) |
| `_adffe` | `SDFFR_X1` | DFF + async reset + enable (SE→enable) |
| `_dffe` | `SDFF_X1` | DFF + enable (SE→enable) |

---

## 4. OpenROAD: `read_verilog` vs `read_db` — BPin 创建

**核心教训:** `read_verilog` + `link_design` 不创建物理 BPin。只有 ORFS 的 `read_db`（从 `.odb` 文件）才能正确初始化 BPin。

```
手动流程 (失败):
  Yosys → netlist.v
  → OpenROAD: read_verilog + link_design
  → BTerm clk 存在，BPins count = 0
  → place_pins → PPL-0021
  → GPL → GPL-0326 "clk not placed" → 全部跳过

ORFS 流程 (正确):
  make do-yosys → .odb (BPins 已初始化)
  → read_db
  → place_pins ✅ → GPL ✅
```

**排查过程:**
1. `place_pin -pin_name clk -location {2500 0}` → PPL-0065（无 BPin 可移动）
2. `place_pin -pin_name clk -layer metal5 -location {2500 0}` → PPL-0022
3. `getFirstPin` 需要 `dbShape&`（C++ API，TCL 不可用）
4. `$bt getBPins` → 0（确认根本原因）
5. `make_tracks` → BPins 仍为 0（不创建 pin shape）
6. `write_def`/`read_def` 回环 → ODB-0251（DEF 缺少 PINS 段）

---

## 5. ORFS 集成：三座大山

### 5.1 Slang 严格编译

ORFS 使用 Slang 前端，比标准 Yosys 更严格：
- `fp32_recip.v` 缺失 → `unknown module` 错误
- `reg` 声明在 `begin` 块中间 → `declaration must come before statements`
- `TL_DATA_WIDTH - SRAM_WIDTH = -64` → `value must be positive`（replication 负数）

### 5.2 两步合成流程

ORFS 需要 canonicalize → synth 两步：
```
do-yosys-canonicalize:  RTL → 1_1_yosys_canonicalize.rtlil
do-yosys:               RTLIL → 1_2_yosys.v + .odb
```

不能跳过 canonicalize，因为 synth.tcl 依赖于 RTLIL 文件。

### 5.3 `keep_hierarchy -min_cost` 与 blackbox 冲突

ORFS 的 `synth.tcl` 中 `keep_hierarchy -min_cost $ungroup_threshold` 会评估所有模块（包括 blackbox）的成本。由于 scratchpad blackbox 无 liberty cell，评估失败。

**修复:** 自定义 synth 脚本，在 `keep_hierarchy -min_cost` 外包 `catch {}`。

---

## 6. Mock→Flat→Surgery 模式（来自 sparsenn）

**适用场景:** 设计中包含 SRAM macro 但综合时需要完整功能验证。

```
1. Mock: SYNTH_MOCK_LARGE_MEMORIES=1 → SRAM 替换为 DFF
2. Flat: synth -flatten → ABC 看到完整逻辑，正确优化
3. Surgery: Python 脚本将 DFF SRAM 替换为 blackbox macro
```

**为什么不能直接用 blackbox:** ABC 看到 blackbox 输出"无用"→ 删除所有读通路逻辑（RocketTile cache 读通路被删→只剩 700 个 DFF）。

---

## 7. FP32 预合成避免 ABC SAT 爆炸

**教训:** 大规模 FP32 运算（数百个 fp32_mul/fp32_add 实例）会导致 ABC SAT solver 爆炸。

**修复:**
1. 每个 FP32 模块单独合成（~17s 每个）
2. 顶层合成用 `read_verilog -lib` 加载为 blackbox
3. 效果：合成时间从 >1h 降至 266s，内存从 5.6GB 降至 1GB

**适用模块:** fp32_mul, fp32_add, fp32_exp, fp32_gelu, fp32_sqrt, pe_int8, mspe_cell

---

## 8. 网表清洗管道（OpenROAD 兼容）

Yosys `write_verilog` 输出需清洗才能被 OpenROAD STA reader 接受：

| 问题 | 修复 |
|------|------|
| 转义标识符 `\$paramod$...` | `$`→`_`, `\` 删除 |
| `reg`/`always`/`assign` | `write_verilog -noexpr -nodec -noattr` |
| 模块参数 `N_CH=s32'000...` | 正则删除 `=s32'...` |
| `initial begin...end` | `memory_map` + `opt -full` 消除 |
| `(* keep_hierarchy *)` | 正则删除 |

**性能:**
- 纯正则，不用逐行循环
- 不用多次 `str.replace`（内存灾难）
- 正则 `re.DOTALL` + `[^)]*` → 嵌套括号无限匹配

---

## 9. GPL 密度与 Die 面积

| 参数 | 建议 |
|------|------|
| PLACE_DENSITY | 0.50-0.55（nangate45） |
| 初始 DIE | 宽松估计，综合后收紧 |
| SRAM 布局 | 3 行结构（减少绕行拥塞） |
| 利用率 | 含 24 个大 SRAM 时不超 60% |

**面积估算教训:** 原始估算非常不准。sparsenn 的 st_block 从估计 12K 实例膨胀到 4.36M 实例（362× 误差）。根本原因：没有考虑 512 通道 × FP32 运算的门级展开。

---

## 10. 工具版本与 API 变更

| 工具 | 版本 | 注意 |
|------|------|------|
| Yosys | 0.64+68 | `write_verilog -noexpr` 仍输出 `wire` 声明 |
| OpenROAD | 26Q1 | `place_pins -random` 已废弃 |
| OpenROAD | 26Q1 | PPL API 变更：`setFirstPinLocation` → 需 `dbShape&` |
| Slang | (ORFS 内置) | 严格 SV 检查，禁止模块引用缺失 |

---

## 11. 调试清单（新设计综合前）

- [ ] 所有子模块输出有消费者（SRAM 写端口 或 顶层端口）
- [ ] 无 `output reg` 直接连 DFF Q 端（BPin 问题）
- [ ] 无 `(expr<<N)[range]` 表达式 part-select
- [ ] 所有 `include` 文件在 ORFS `SRC_DIR` 中
- [ ] `fp32_recip` 等依赖模块完整
- [ ] `TL_DATA_WIDTH - SRAM_WIDTH` 等常量表达式非负
- [ ] `reg` 声明在 `begin` 块顶部的 Verilog-2001 约束
- [ ] dfflibmap 输出全部是目标库 cell（检查 `grep -c "_adff\|_dffe"`）
- [ ] 黑盒模块在 `SYNTH_BLACKBOXES` 中列出
- [ ] `keep_hierarchy -min_cost` 对黑盒安全（自定义 synth 脚本）
- [ ] ORFS 文件在 `designs/src/` 而非 `src/`（DESIGN_HOME 路径）
- [ ] Docker 容器对 `results/` 目录有写权限

---

## 12. ORFS 两步合成：canonicalize + synth 分离

**教训:** ORFS 合成分两步：`synth_canonicalize.tcl`（RTL→RTLIL）和 `synth.tcl`（RTLIL→mapped netlist）。二者分离允许增量迭代——RTL 不变时 canonicalize 可缓存。

**关键点：**
- `do-yosys-canonicalize` 产 `1_1_yosys_canonicalize.rtlil`（94MB+）
- `do-yosys` 依赖 RTLIL 文件存在
- 修改 RTL 后必须清 RTLIL 重跑 canonicalize
- 只改 config/synth 脚本时 RTLIL 可复用

## 13. `keep_hierarchy -min_cost` 与 Blackbox 冲突

**教训:** ORFS `synth.tcl` 中的 `keep_hierarchy -min_cost $ungroup_threshold` 会评估所有模块的成本。Blackbox 无 liberty cell → 评估失败 → 整个 synth 失败。

**解决：** 自定义 synth 脚本直接删除 `keep_hierarchy -min_cost` 行。`setattr keep_hierarchy=1` 和 `catch {}` 均无效——Yosys pass 错误无法被 TCL catch 捕获。

## 14. Slang 严格编译 vs read_verilog -sv

**教训:** ORFS 使用 Slang 前端（`SYNTH_HDL_FRONTEND=slang`），比 Yosys `read_verilog -sv` 严格得多。

**Slang 额外检查：**
| 问题 | 示例 | 修复 |
|------|------|------|
| 声明顺序 | `begin` 块中 `reg` 必须在语句前 | 移到函数顶部 |
| 模块引用 | 所有实例化模块必须在编译列表 | 补全 `fp32_recip.v` |
| 常量表达式 | `(64-128){1'b0}` → 负数 replication | 简化为 `dma_acc_rd_raw[63:0]` |

## 15. ORFS 路径规范

**教训:** ORFS 的 `DESIGN_HOME` 指向 `designs/`（不是项目根目录）。

| 变量 | 实际路径 |
|------|---------|
| `DESIGN_HOME` | `designs/` |
| `SRC_DIR` | `designs/src/<design>/` |
| `ADDITIONAL_LEFS` | 需完整相对路径 |

RTL 文件必须放在 `designs/src/<design>/`，而非 `src/<design>/`。

## 16. Blackbox LEF 端口定义

**教训:** `bb_stubs.lef` 中的 PIN 定义必须与实际模块端口匹配。`PIN dmy` 通配符在 `link_design` 时导致端口不匹配警告（STA-0201），虽不致命但产生大量噪声。

**修复：** 为每个 blackbox 定义正确的端口名（如 `fp32_mul` 需要 PIN `a`, `b`, `result`）。

## 17. GPL 密度与 Die 面积的平衡

**教训:** GPL 密度过高（>55%）导致迭代发散和大量重叠。

| 密度 | 结果 |
|------|------|
| 0.35-0.45 | 收敛稳定，~7-9% 重叠 |
| 0.50-0.55 | 边缘，可能发散 |
| >0.62 | 发散，GPL 回退到历史快照 |

**Die 压缩策略：** 从 6400→4500→4200→4000 迭代压缩，4200 以下 GPL 发散。4500 为最优平衡点（47% util，无发散）。

## 18. GPL `-skip_io` 与 Clock BTerm

**教训:** 手动 `read_verilog`+`link_design` 流程在 OpenROAD 26Q1 中不创建物理 BPin。即使是 ORFS `do-1_synth` 的 `synth_odb.tcl`→`load_design` 也同样没有 BPin。

**尝试过的方法：**
| 方法 | 结果 |
|------|------|
| `place_pin -pin_name clk` | PPL-0065 |
| ODB API `$bt setLocation` | 方法不存在 |
| ODB API `getFirstPin` | 需 C++ `dbShape&` 参数 |
| `place_pins -annealing` | PPL-0021（macro 未放置） |
| DEF PINS 注入 + read_def | ODB-0251 |
| 先 floorplan 后 link_design | "no network linked" |

**最终方案：** `global_placement -skip_io` 绕过检查，接受 ~9% 重叠（无 detailed placement 合法化）。

## 19. RTL 变更后必须全流程验证

**教训:** 删除 INT8 阵列后 RTL 有 3 个新问题（Slang 声明顺序、缺失 fp32_recip、负数 replication），每个都导致 canonicalize 失败。修复后 canonicalize 才通过。永远不要假设小改动不需要重新验证。

## 20. 备份与可复现性

**教训:** 大型 DEF 文件（60MB+）是唯一不可替代的产出。必须与渲染脚本一起单独备份。

**备份清单：**
```
backup/
├── dsa_placed_orfs.def       ← 核心：GPL placement 数据
├── render_modules.py          ← DSA 热力图脚本
├── render_soc_heatmap.py      ← SoC 热力图脚本
├── dsa_placement_heatmap.md   ← 面积数据
└── README.md                  ← 复现步骤
```
