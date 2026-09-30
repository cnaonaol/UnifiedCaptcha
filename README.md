# Qwen2.5-VL
- 2025.01.28: 先是发布了Qwen2.5-VL 3B、7B、72B
- 2025.03.25: 发布了Qwen2.5-VL 32B, 使用强化学习进一步优化

# SFT
## 数据
### Grounding数据
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "<image>描述图像"
    },
    {
      "role": "assistant",
      "content": "<ref-object><bbox>和<ref-object><bbox>正在沙滩上玩耍"
    }
  ],
  "images": [
    "/xxx/x.jpg"
  ],
  "objects": {
    "ref": [
      "一只狗",
      "一个女人"
    ],
    "bbox": [
      [
        331.5,
        761.4,
        853.5,
        1594.8
      ],
      [
        676.5,
        685.8,
        1099.5,
        1427.4
      ]
    ]
  }
}
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "<image>找到图像中的<ref-object>"
    },
    {
      "role": "assistant",
      "content": "<bbox><bbox>"
    }
  ],
  "images": [
    "/xxx/x.jpg"
  ],
  "objects": {
    "ref": [
      "羊"
    ],
    "bbox": [
      [
        90.9,
        160.8,
        135,
        212.8
      ],
      [
        360.9,
        480.8,
        495,
        532.8
      ]
    ]
  }
}
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "<image>帮我打开谷歌浏览器"
    },
    {
      "role": "assistant",
      "content": "Action: click(start_box='<bbox>')"
    }
  ],
  "images": [
    "/xxx/x.jpg"
  ],
  "objects": {
    "ref": [],
    "bbox": [
      [
        615,
        226
      ]
    ]
  }
}
```

## 训练
### 单卡训练
```shell
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'AI-ModelScope/coco#2000' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
```