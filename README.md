# Spatial Lite Benchmark：样本生成与评测脚本
 
本目录是 SpatiaLite 基准的一个子模块，用于在不同任务上**批量生成模型回答并进行评测**。核心脚本为 `generate_response_samples_new.py`，通过命令行接口调用大模型，对 Mental Rotation、Cube Rolling、Rubik’s Cube、Moving Box、Wood Slide 等任务进行统一评测。

---

## 1. 论文简介（SpatiaLite & IDF 框架）

SpatiaLite 来自论文：

> **Imagine-in-Space: Exploring the Frontier of Spatial Intelligence and Reasoning Efficiency in Vision–Language Models**  
> Anonymous submission

论文主要观点简要概括如下：

- **问题背景**：当前先进 VLM（如 DeepSeek-R1、OpenAI o3、Gemini 2.5 Pro）在逻辑推理和数学问题上表现出很强能力，但在**空间推理**（心智旋转、空间导航、空间关系理解等）上仍明显不足。
- **核心假设**：空间世界模型（Spatial World Model）中，**“想象（imagination）”是主导机制**，可以分为两种形式：
  - 语言想象（Linguistic imagery）：用语言/符号表达空间状态并在语言空间中推理；
  - 视觉想象（Visual imagery）：在视觉表征空间中进行感知、预测与决策，更接近人类的心像。
- **SpatiaLite 基准**：构建了一个**完全合成（synthetic）**的空间推理基准，包含五类任务：
  - **Visual-Centric**：Mental Rotation；
  - **Linguistic-Centric**：Cube Rolling、Rubik’s Cube；
  - **Collaborative Spatial Reasoning**：Moving Box（推箱子）、Wood Slide（华容道）。
  该基准同时评测**准确率**与**推理效率（token 消耗）**。
- **主要发现**：
  - 先进 VLM 在需要真实视觉-空间表征与 3D 几何变换的任务（如 Mental Rotation）上远落后于人类；
  - 在 Moving Box / Wood Slide 等需要长序列规划的任务中，模型往往通过极长的语言链式推理「硬算」，导致**严重的推理效率低下**（token 随复杂度指数级增长）；
  - Gemini 2.5 Pro 在带视觉输入的协同空间任务上可以利用图像触发更直观的视觉驱动策略，在准确率和效率上优于单纯语言输入。
- **IDF（Imagery-Driven Framework）**：
  - 提出一个两阶段的 Imagery-Driven Framework（IDF），通过大规模视觉心像数据合成与推理蒸馏，帮助 VLM **隐式构建内部空间世界模型**；
  - 在 Cube Rolling、Rubik’s Cube 等任务上明显提升空间变换建模与预测能力。

如果你使用 SpatiaLite 或 IDF 相关代码，请在论文正式公开后，采用论文中推荐的引用格式进行引用。

---

## 2. 脚本功能概览

`generate_response_samples_new.py` 提供三个核心能力：

- **样本生成**：
  - 调用指定大模型，为数据集中每个样本生成回答（描述类答案或执行序列）；
- **离线评测**：
  - 从已有 json/jsonl 文件中读取模型输出，统一用 LLM/环境模拟进行打分；
- **失败分析（可选）**：
  - 对失败样本进行错误类型归因（感知层 / 变换层 / 策略层）。

脚本入口采用 [Python Fire](https://github.com/google/python-fire)，可直接通过命令行调用。

---

## 3. 模型配置（必须完成）

在使用本脚本前，需要在项目中完成 **模型选择与 API 配置**：

- 在 `modeling.py` 中实现：
  - `select_model(model_name=..., **kwargs)`，返回一个带有 `run(prompt, image=None)` 方法的模型封装；
  - 该封装内部负责调用实际的后端服务（OpenAI / Azure / DeepSeek / Gemini 等）。
- 需要在本地环境变量或配置文件中设置好：
  - 各模型对应的 API Key、Endpoint、模型名称等；
  - 保证你在命令中传入的 `--model_name` 能被 `select_model` 正确识别。

> 一句话：你只需要保证 `select_model(**kwargs)` 能根据 `--model_name` 返回一个可调用的大模型对象，本脚本就可以直接复用。

---

## 4. 数据准备（最小要求）

- 数据集按任务组织成若干 JSON 文件，例如：
  - `/data/move_box/move_box_samples.json`
  - `/data/rubiks_cube/rubiks_cube_colors.json`
- 每条样本通常包含：
  - 图像路径或编码（例如 `image` 或 `image_string`）；
  - 问题文本（如 `question` 或其他字段）；
  - 标准答案（如 `answer`，执行式任务可能还包含 `environment_type`、`initial_state` 等）。
- 具体字段结构由 `data_loading.Data` 和相关样本类定义，本 README 不再展开，保持实现细节在代码中。

---

## 5. 最小评测流程

下面给出一个「从模型生成到评测」的最小闭环示例，你可以按需改参数：

### 5.1 生成模型回答（可选）

```bash
python generate_response_samples_new.py batch_generate_samples \
  --dataset /data/move_box \
  --task_name move_box_samples \
  --question_type executive \
  --model_name o4-mini \
  --output_dir outputs_spatialite
```

- `--dataset`：数据目录，内部包含 `<task_name>.json` 和图像文件（或上级有 `data/` 子目录）。
- `--task_name`：JSON 文件名（不含 `.json`）。
- `--question_type`：`descrip`（描述类）或 `executive`（执行类）。
- `--model_name`：与 `select_model` 中的配置保持一致。
- 输出：在 `outputs_spatialite/...` 下生成带时间戳的 jsonl 文件，记录每条样本的模型输出。

如果你已经有现成的模型输出文件，也可以跳过这一步，直接进行评测。

### 5.2 基于已有结果离线评测

```bash
python generate_response_samples_new.py evaluate_from_file \
  --input_file "outputs_spatialite/data/move_box/executive/o4-mini/move_box_samples_*.jsonl" \
  --output_dir "outputs_spatialite/eval_move_box" \
  --question_type "executive"
```

- `evaluate_from_file` 会：
  - 加载 `input_file` 中的样本与模型预测；
  - 对 `descrip` 任务使用 LLM 打 0/1 分；
  - 对 `executive` 任务解析执行序列，在仿真环境中执行并判定成功与否；
  - 统计准确率等指标，并输出带统计信息的 JSON 结果。

### 5.3 失败样本分析（可选）

```bash
python generate_response_samples_new.py analyze_failures_from_file \
  --input_file "outputs_spatialite/eval_move_box/eval_xxx.jsonl" \
  --output_dir "outputs_spatialite/analysis_move_box" \
  --sample_rate 0.5
```

- 仅对评分为特定失败类型（如解析失败）的样本进行采样分析；
- 输出每条失败样本的错误类型（感知 / 变换 / 策略）与对应解释，辅助论文中的误差分析与可视化。

---

## 6. 引用说明

如果你在论文或项目中使用了 SpatiaLite 基准、IDF 框架或本评测代码，请在 SpatiaLite 论文正式公开后：

- 在参考文献中引用：
  - *Imagine-in-Space: Exploring the Frontier of Spatial Intelligence and Reasoning Efficiency in Vision–Language Models*；
- 并在合适位置说明：
  - 评测管线基于 SpatiaLite 提供的官方实现（本仓库中的 `generate_response_samples_new.py`）。

当论文提供正式 BibTeX 后，你可以将其直接补充到本 README 中.
