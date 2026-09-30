# QwenVL 训练框架

本仓库为 Qwen VL 模型提供了一个训练框架。使用我们的仓库分为两个步骤：

1. 自定义你的数据集：下载数据，实现配置
2. 修改训练脚本：

## 仓库结构

`qwenvl` 目录包含以下组件：

### `train/`

* `trainer.py`：基于 Huggingface Trainer 更新的主要 Trainer
* `train_qwen.py`：训练的主文件
* `argument.py`：模型、数据和训练参数的 Dataclass

### `data/`

* `__init__.py`：包含数据集配置
* `data_processor.py`：用于 QwenVL 模型的数据处理模块
* `rope2d.py`：提供 RoPE 实现

### `tools`

* `process_bbox.ipynb`：将 bbox 转换为 QwenVL 格式。如果你有 grounding 数据，请参考此文件转换你的数据。
* `pack_data.py`：将数据打包到长度均匀的 bucket 中。

## 环境要求

你可以使用以下版本的软件包：

* `torch==2.6.0`
* `torchvision==0.21.0`
* `transformers==4.57.0.dev0`
* `deepspeed==0.17.1`
* `flash_attn==2.7.4.post1`
* `triton==3.2.0`
* `accelerate==1.7.0`
* `torchcodec==0.2`
* `peft==0.17.1`

## 自定义数据集配置

自定义数据应具有如下格式：

### JSON 数据结构

**媒体说明**：

* `image/video`：包含媒体文件的路径（必需）
* prompt 中的媒体标签：

  * `<image>` 用于图像理解任务
  * `<video>` 用于视频理解任务
* `conversations`：包含问题和答案

### 示例数据：

1. **单图像示例**：

```json
{
    "image": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n这张图片中的主要物体是什么？"
        },
        {
            "from": "gpt",
            "value": "木桌上的一个红苹果"
        }
    ]
}
```

2. **多图像示例**：

```json
{
    "image": ["cats/001.jpg", "cats/002.jpg"],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<image>\n这两只猫有什么区别？"
        },
        {
            "from": "gpt",
            "value": "第一只猫是一只橙色虎斑猫，短毛、绿眼睛，而第二只是一只灰色暹罗猫，蓝眼睛并带有重点色。它们看起来也处于不同的环境中——第一只在室内的沙发上，第二只在室外的花园里。"
        }
    ]
}
```

3. **视频示例**：

```json
{
    "video": "videos/005.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\n是什么导致蓝色物体移动？\n选项：\n(A) 重力\n(B) 碰撞\n(C) 磁力"
        },
        {
            "from": "gpt",
            "value": "答案：(B) 碰撞"
        }
    ]
}
```

4. **Grounding 示例**：

```json
{
    "image": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n定位图像中的房子，并以 JSON 格式输出 bbox 坐标。"
        },
        {
            "from": "gpt",
            "value": "{\n\"bbox_2d\": [135, 114, 1016, 672]\n}"
        }
    ]
}
```

5. **打包数据示例**：

```json
[
    {
        "image": "images/001.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n这张图片中的主要物体是什么？"
            },
            {
                "from": "gpt",
                "value": "木桌上的一个红苹果"
            }
        ]
    },
    {
        "image": "images/002.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n这张图片中的主要物体是什么？"
            },
            {
                "from": "gpt",
                "value": "塑料桌上的一个绿色橙子"
            }
        ]
    }
]
```

一些示例位于 `demo/single_images.json` 和 `demo/video.json` 中，这些 json 文件可以用于训练。

### 用于训练的数据集配置

要添加或修改训练数据集，请按照以下步骤操作：

### 数据集定义结构

1. **创建一个数据集字典**，格式与 `data/__init__.py` 文件中一致：

```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # 如果路径已经位于 annotation 中，则可以为空
}
```

2. **注册你的数据集**，将其添加到 `data_dict` 中：

```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... 其他数据集
}
```

### 采样比例控制

你可以选择通过在数据集名称后附加 `%X` 来指定采样比例：

* `"dataset_name%50"` 将采样 50% 的数据
* `"dataset_name%20"` 将采样 20% 的数据

### 使用示例

1. 定义你的数据集：

```python
MY_DATASET = {
    "annotation_path": "/data/my_dataset/annotations.json",
    "data_path": "/data/my_dataset/images/",
}

data_dict = {
    "my_dataset": MY_DATASET,
    "cambrian_737k": CAMBRIAN_737K,  # 已有数据集
}
```

2. 在训练中使用：

```python
dataset_names = ["my_dataset%50"]  # 将使用你数据集的 50%
configs = data_list(dataset_names)
```

### 注意事项

* `annotation_path` 应指向一个包含数据集标注的 JSON 或 JSONL 文件。
* 如果 annotation 中的图像路径为绝对路径，则 `data_path` 可以留空。
* 当指定多个数据集时，采样比例会分别应用于每个数据集。
* 一些可以直接使用的数据集：`nyu-visionx/Cambrian-10M`、`lmms-lab/LLaVA-NeXT-Data`、`FreedomIntelligence/ALLaVA-4V`、`TIGER-Lab/VisualWebInstruct`。
* 训练数据应严格遵循以下格式：

  * 问题中的一个 `<image>` 标签必须且只能对应一个图像文件
  * 同样，`<video>` 标签必须对应视频文件
  * 这些特殊 token 不应出现在答案文本中
* 对于可能存在图像缺失或其他问题的开源数据，你可以使用 `tools/check_image.py` 验证数据完整性。

## 使用方法

训练模型：

```bash
#!/bin/bash
# 完整的 QwenVL 训练启动脚本，包含完整参数说明

# ======================
# 分布式配置
# ======================
MASTER_ADDR="127.0.0.1"                     # [必需] 多 GPU 训练的主节点 IP
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # 随机端口，用于避免冲突
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # 自动检测可用 GPU

# ======================
# 路径配置
# ======================
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] 预训练模型路径
OUTPUT_DIR="./checkpoints"                   # 保存 checkpoint 的目录
CACHE_DIR="./cache"                          # [TrainingArguments] 模型缓存目录

# ======================
# 模型配置
# ======================
DATASETS="your_dataset%100"                  # [DataArguments] 带采样比例的数据集

# ======================
# 训练超参数
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         # 核心参数
         --model_name_or_path $MODEL_PATH \  # [ModelArguments] 模型标识
         --tune_mm_llm True \                # [TrainingArguments] 是否训练 LLM
         --tune_mm_vision False \            # [TrainingArguments] 是否训练 VIT
         --tune_mm_mlp False \               # [TrainingArguments] 是否训练 MLP
         --dataset_use $DATASETS \           # [DataArguments] 数据集配置
         --output_dir $OUTPUT_DIR \          # checkpoint 输出目录
         --cache_dir $CACHE_DIR \            # [TrainingArguments] 模型缓存位置
         
         # 精度与显存
         --bf16 \                            # 使用 bfloat16 精度（Ampere 及以上 GPU）
         --per_device_train_batch_size 4 \   # 每张 GPU 的 batch size
         --gradient_accumulation_steps 4 \   # 有效 batch size 倍增系数
         
         # 学习率配置
         --learning_rate 2e-7 \              # 基础学习率
         --mm_projector_lr 1e-5 \            # [TrainingArguments] Projector 专用学习率
         --vision_tower_lr 1e-6 \            # [TrainingArguments] 视觉编码器学习率
         --optim adamw_torch \               # [TrainingArguments] 优化器选择
         
         # 序列配置
         --model_max_length 4096 \           # [TrainingArguments] 最大序列长度
         --data_flatten True \               # [DataArguments] 拼接 batch 中的序列
         --data_packing True \               # [DataArguments] 使用打包数据
         
         # 图像处理
         --max_pixels 576\*28\*28 \          # [DataArguments] 图像最大像素数（H*W）
         --min_pixels 16\*28\*28 \           # [DataArguments] 图像最小像素数
         # 视频处理
         --video_fps 2 \                     # [DataArguments] 视频 fps
         --video_max_frames 8 \              # [DataArguments] 每个视频的最大帧数
         --video_min_frames 4 \              # [DataArguments] 每个视频的最小帧数
         --video_max_pixels 1664\*28\*28 \   # [DataArguments] 每个视频的最大像素数
         --video_min_pixels 256\*28\*28 \    # [DataArguments] 每个视频的最小像素数
         
         # 训练计划
         --num_train_epochs 3 \              # 总训练 epoch 数
         --warmup_ratio 0.03 \               # 学习率 warmup 比例
         --lr_scheduler_type "cosine" \      # 学习率调度策略
         --weight_decay 0.01 \               # L2 正则化强度
         
         # 日志与 Checkpoint
         --logging_steps 10 \                # 日志指标记录间隔
         --save_steps 500 \                  # Checkpoint 保存间隔
         --save_total_limit 3 \              # 最大保留 checkpoint 数

         # LoRA 配置
         --lora_enable True \                # [TrainingArguments] 启用 LoRA
         --lora_r 8 \                        # [TrainingArguments] LoRA r
         --lora_alpha 16 \                   # [TrainingArguments] LoRA alpha
         --lora_dropout 0.0 \                # [TrainingArguments] LoRA dropout

         # 高级选项
         --deepspeed zero3.json \            # DeepSpeed 配置
```

该脚本接受三类参数：

* 用于控制训练哪些组件的参数（`tune_mm_vision`、`
