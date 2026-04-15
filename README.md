# forward_surrogate_data_gen

`forward_surrogate_data_gen` 是一个独立的数据生成层，专门负责把 `bsp2fe` 的有限元求解结果整理成可用于正向代理模型训练的标准样本。
'control_point_sampling' 用于对原始控制点进行固定化采样，以此统一格式

整个项目的大致流程如下

- 输入一组 B-spline 曲面控制点文件 `Surface-*.npz`
- 调用 `bsp2fe` 完成充气有限元求解
- 在原始 B-spline 曲面上固定采样 `64 x 64` 个曲面点
- 调用`forward_surrogate_data_gen`把 FE 表面位移传递到这些固定采样点上
- 导出统一格式的 `.npz` 样本和 `manifest.jsonl`

---

## 1. 项目

当前整体方案被拆成了两个项目：

- `bsp2fe`
  - 只负责 B-spline 到有限元模型的核心建模与求解
  - 负责几何转 STEP、Gmsh 网格化、torchfea 组装与压力求解
- `forward_surrogate_data_gen`
  - 只负责训练数据生成
  - 负责固定采样、位移传递、样本导出、批量运行与结果清单


## 2. 整体流程

整个数据生成流程如下：

```text
Surface-*.npz
-> 按 case 分组
-> 读取为 bspmap.BSP 曲面对象
-> 检查 v 方向首层/末层控制点是否共面
-> 调用 bsp2fe.build_parametric_pneumatic_model(...)
-> torchfea 求解充气后的 FE 结果
-> 提取目标内气腔表面的 FE 静止/变形网格
-> 在原始 B-spline 曲面上固定采样 64x64 个点
-> 将 FE 位移传递到这 64x64 个采样点
-> 构造 inner_rest_surface / inner_def_surface / inner_disp_field
-> 导出 .npz 样本、manifest.jsonl、summary.json
```



## 3. 输入数据要求

### 3.1 文件命名规则

输入目录下需要放置一组或多组 `Surface-*.npz` 文件，命名规则示例：

```text
Surface-0_iter-1.npz
Surface-1_iter-1.npz
Surface-0_iter-2.npz
Surface-1_iter-2.npz
```

其中：

- `Surface-0_*` 表示外壳曲面
- `Surface-1_*` 表示第 1 个内气腔曲面
- `Surface-2_*`、`Surface-3_*` 等可表示更多内气腔
- 相同后缀 token 的文件会被视为同一个 case

例如：

- `Surface-0_iter-1.npz`
- `Surface-1_iter-1.npz`

会被自动配对成同一个样本 case。

### 3.2 曲面语义约定

在所有 case 中，曲面顺序必须满足：

- `surfaces[0]`：外壳 / outer shell
- `surfaces[1:]`：内气腔 / cavity surfaces

当命令行中设置：

```text
--target-cavity-index 1
```

表示导出 `surfaces[1]` 这个内气腔的位移场样本。


## 4. 64x64 采样点

它们是：

- 先由原始控制点定义出连续的 B-spline 曲面
- 再在该曲面的参数空间 `(u, v)` 中均匀取一个固定网格
- 最后通过曲面映射得到真实三维空间中的曲面采样点

也就是说，这 64x64 个点是：

- 原始 B-spline 曲面上的固定参数采样点
- 用来统一表示不同几何 case 的规则曲面网格


## 5. `bsp2fe` 是如何被调用的

### 5.1 调用入口

批量命令入口是：

- `src/forward_surrogate_data_gen/generate_dataset.py`

它会解析命令行参数，然后调用：

- `run_forward_dataset_export(...)`

### 5.2 批量导出入口

`run_forward_dataset_export(...)` 会做这些事：

1. 扫描输入目录下所有 `Surface-*.npz`
2. 按 case 分组
3. 逐个 case 读取为 `bspmap.BSP`
4. 对每个 pressure 构造一次求解任务
5. 调用 `generate_forward_surrogate_sample(...)`

### 5.3 单样本求解入口

`generate_forward_surrogate_sample(...)` 中会直接调用：

- `bsp2fe.build_parametric_pneumatic_model(...)`

这个函数来自 `bsp2fe`，会完成：

- B-spline 曲面转 STEP
- Gmsh 网格化
- `.inp` 读回
- torchfea 装配
- NeoHookean 材料设置
- 腔体压力载荷设置

之后再执行：

- `model.fe.solve()`

得到 FE 求解结果。

所以当前关系是：

```text
forward_surrogate_data_gen
    └── 调用 bsp2fe.build_parametric_pneumatic_model(...)
            └── 内部完成 gmsh + torchfea 的 FE 求解
```

---

## 6. 从 FE 到训练样本是怎么构造的

单个样本的生成分成 4 步：

### 第 1 步：保留原始控制点

导出时会先保存原始控制点：

- `inner_ctrl_raw`
- `outer_ctrl_raw`


### 第 2 步：在原始曲面上固定采样

在 inner / outer 的原始 B-spline 曲面上分别采样：

- `inner_surface_samples`
- `outer_surface_samples`

默认采样分辨率是：

- `64 x 64`

### 第 3 步：从 FE 中提取目标内气腔表面

FE 求解完成后，会提取目标内气腔表面的：

- 静止状态表面网格
- 变形后表面网格

如果两者三角拓扑不一致，程序会直接报错，因为无法稳定地传递位移。

### 第 4 步：把 FE 位移传递到固定采样点

对于每个 `inner_surface_samples` 上的采样点：

1. 在 FE 静止表面上找到最近三角形
2. 取最近点作为 FE 对齐参考点
3. 在该静止三角形上计算重心坐标
4. 使用相同重心权重插值变形后三角形位置
5. 得到采样点位移

最终构造：

- `inner_rest_surface`
- `inner_def_surface`
- `inner_disp_field`

其中：

```python
inner_disp_field = inner_def_surface - inner_rest_surface
```

当前实现里：

- `inner_rest_surface` 使用的是原始 B-spline 曲面采样点
- `inner_fe_aligned_rest_surface` 使用的是 FE 表面上的最近点

这样可以同时保留：

- 原始几何参数化的一致性
- 与 FE 表面的对齐信息

---

## 7. 导出样本字段说明

每个导出的 `.npz` 样本至少包含以下字段。

### 7.1 几何来源字段

- `inner_ctrl_raw`
  - 原始内气腔控制点
  - shape 不固定，例如 `(74, 24, 3)`
- `outer_ctrl_raw`
  - 原始外壳控制点
  - shape 不固定

### 7.2 固定采样曲面字段

- `inner_surface_samples`
  - 从原始内气腔 B-spline 曲面上采样得到的固定 `64 x 64` 曲面点
  - shape = `(64, 64, 3)`
- `outer_surface_samples`
  - 从原始外壳 B-spline 曲面上采样得到的固定 `64 x 64` 曲面点
  - shape = `(64, 64, 3)`
- `surface_uv_grid`
  - 对应采样点的参数坐标 `(u, v)` 网格
  - shape = `(64, 64, 2)`

### 7.3 载荷字段

- `pressure`
  - 当前目标腔体施加的压力标量
  - shape = `(1,)`
- `pressure_values_all_cavities`
  - 当前 FE 求解时所有腔体的压力向量
  - shape = `(num_cavities,)`

### 7.4 输出监督字段

- `inner_rest_surface`
  - 内气腔固定采样点在静止状态下的三维位置
  - shape = `(64, 64, 3)`
- `inner_def_surface`
  - 同一批采样点在充气变形后的三维位置
  - shape = `(64, 64, 3)`
- `inner_disp_field`
  - 固定采样点上的三维位移场
  - shape = `(64, 64, 3)`

定义为：

```python
inner_disp_field = inner_def_surface - inner_rest_surface
```

### 7.5 FE 对齐字段

- `inner_fe_aligned_rest_surface`
  - 每个采样点在 FE 静止表面上的最近点
  - shape = `(64, 64, 3)`
- `inner_fe_aligned_def_surface`
  - 使用同一 FE 三角形重心坐标传递后的变形点
  - shape = `(64, 64, 3)`

### 7.6 可选 FE 原始网格字段

默认情况下还会保存：

- `inner_fe_rest_points`
- `inner_fe_rest_triangles`
- `inner_fe_def_points`
- `inner_fe_def_triangles`

若不需要，可在命令行加：

```bash
--exclude-raw-fe-mesh
```

### 7.7 元信息字段

导出的 `.npz` 中还会保存：

- `case_key`
- `target_surface_name`
- `target_cavity_index`
- `sample_grid_shape`
- `inner_ctrl_raw_shape`
- `outer_ctrl_raw_shape`
- `surface_validation_json`
- `metadata_json`


## 8. 输出目录结构

运行一次批量导出后，输出目录大致如下：

```text
output-root/
├─ samples/
│  ├─ <case>__cavity1__p_0p020000.npz
│  ├─ <case>__cavity1__p_0p040000.npz
│  └─ ...
├─ work/
│  ├─ <case>__cavity1__p_0p020000/
│  └─ ...
├─ manifest.jsonl
├─ failures.jsonl
└─ summary.json
```

各文件含义如下：

- `samples/`
  - 导出的训练样本
- `work/`
  - 每个样本运行时的临时 FE 工作目录
- `manifest.jsonl`
  - 成功或跳过的样本清单
- `failures.jsonl`
  - 失败样本清单和错误信息
- `summary.json`
  - 本次导出的整体统计摘要

---

## 9. v 方向首尾层共面约束

按照当前几何约定，会检查每张 B-spline 曲面是否满足：

- v 方向第一层控制点
- v 方向最后一层控制点

位于同一个平面内。

### 9.1 默认行为

默认情况下：

- 会检查
- 会把结果写入元数据
- 但不会阻塞样本导出

### 9.2 严格模式

如果希望严格执行这条规则，可以加：

```bash
--enforce-v-plane
```

此时如果不满足共面要求，导出会直接失败，并记录到 `failures.jsonl`。

### 9.3 容差设置

默认容差是：

```text
1e-6
```

也可以通过命令行调整：

```bash
--v-plane-tol 1e-5
```

---

## 10. 运行方式

### 10.1 基本命令

```bash
cd "C:\Users\broo\Desktop\AI for SoRo Design\forward_surrogate_data_gen"
D:\Programs\Python\Python312\python.exe src\forward_surrogate_data_gen\generate_dataset.py ^
  --input-root "C:\Users\broo\Desktop\AI for SoRo Design\geometry\geometry" ^
  --output-root "C:\Users\broo\Desktop\AI for SoRo Design\forward_surrogate_data_gen\outputs\dataset_run" ^
  --pressures 0.02 0.04 0.06 ^
  --target-cavity-index 1 ^
  --mesh-size 1.2 ^
  --mu 0.48 ^
  --kappa 4.82 ^
  --sample-grid-height 64 ^
  --sample-grid-width 64
```

### 10.2 常用参数说明

- `--input-root`
  - 输入曲面文件目录
- `--output-root`
  - 输出样本目录
- `--pressures`
  - 目标腔体要扫描的压力值列表
- `--target-cavity-index`
  - 目标内气腔编号，从 1 开始
- `--mesh-size`
  - Gmsh 最大网格尺寸
- `--mu`
  - NeoHookean 材料参数 `mu`
- `--kappa`
  - NeoHookean 材料参数 `kappa`
- `--solver-tol-error`
  - torchfea 非线性求解误差容差
- `--sample-grid-height`
  - 采样网格高
- `--sample-grid-width`
  - 采样网格宽
- `--exclude-raw-fe-mesh`
  - 不保存原始 FE 表面网格
- `--overwrite`
  - 即使样本已存在也重新计算
- `--enforce-v-plane`
  - 严格要求 v 向首尾层共面

---
## 11. 采样方法

当前控制点采样采用：

- 在原始控制点网格索引空间上做规则重采样
- 采样方法为双线性插值

### 11.1 输入

原始控制点网格：

- `ctrl_raw.shape = (H_raw, W_raw, 3)`

其中：

- 第 1 维和第 2 维保持原始控制点排列顺序
- 第 3 维是三维坐标 `(x, y, z)`

### 11.2 输出

采样后控制点网格：

- `ctrl_sampled.shape = (64, 64, 3)`

### 11.3 具体做法

对原始控制点网格的两个索引轴分别做线性映射：

- 高度方向：`[0, H_raw - 1] -> [0, 63]`
- 宽度方向：`[0, W_raw - 1] -> [0, 63]`

然后在原始控制点网格上做双线性插值，得到新的固定控制点网格。

### 11.4 边界保持

- 左上角对应原始左上角
- 右上角对应原始右上角
- 左下角对应原始左下角
- 右下角对应原始右下角

因此整体几何顺序不会乱。

---



## 12. 采样输出


### 12.1 说明性样本

保存在：

- `output-root/samples/`

主要字段包括：

- `inner_ctrl_raw`
- `outer_ctrl_raw`
- `inner_ctrl_sampled`
- `outer_ctrl_sampled`
- `control_grid_shape`
- `surface_validation_json`


### 12.2 可直接给 FE 脚本使用的曲面文件

保存在：

- `output-root/surface_files/`

文件名仍然保持为：

- `Surface-0_*.npz`
- `Surface-1_*.npz`

并且使用：

- `bspmap.BSP.save()`

保存成和原始 `Surface-*.npz` 相同的格式

---

## 13.3. 采样运行脚本

当前独立脚本：

- `src/control_point_sampling/sample_control_points.py`

运行方式：

```bash
cd "C:\Users\broo\Desktop\AI for SoRo Design\control_point_sampling"
python src\control_point_sampling\sample_control_points.py ^
  --input-root "C:\Users\broo\Desktop\AI for SoRo Design\geometry\geometry" ^
  --output-root "C:\Users\broo\Desktop\AI for SoRo Design\control_point_sampling\outputs\control_point_samples" ^
  --target-cavity-index 1 ^
  --control-grid-height 64 ^
  --control-grid-width 64
```





