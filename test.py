# diagnose.py
import torch
import numpy as np
import os

# 检查模型权重文件
weights_path = 'models/deploy_gru_weights.pth'
if os.path.exists(weights_path):
    print(f"✅ 找到权重文件: {weights_path}")

    # 加载权重
    state_dict = torch.load(weights_path, map_location='cpu')

    print(f"\n权重文件中的键和形状:")
    for key, value in state_dict.items():
        print(f"  {key}: {value.shape}")

    # 推断模型参数
    if 'rnn.weight_ih_l0' in state_dict:
        ih_shape = state_dict['rnn.weight_ih_l0'].shape
        print(f"\n推断的模型参数:")
        print(f"  rnn.weight_ih_l0形状: {ih_shape}")

        # hidden_size = ih_shape[0] / 4
        hidden_size = ih_shape[0] // 4
        input_size = ih_shape[1]
        print(f"  hidden_size = {hidden_size}")
        print(f"  input_size = {input_size}")

        # 检查是否有第二层
        if 'rnn.weight_ih_l1' in state_dict:
            print(f"  num_layers >= 2")
else:
    print(f"❌ 找不到权重文件: {weights_path}")
    print("请先运行 train_model.py 训练模型")