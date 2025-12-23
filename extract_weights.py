import torch

# 加载训练保存的 checkpoint（含优化器）
# ckpt = torch.load('workspace/nanodet-plus-m_416_Feng/model_best/nanodet_model_best.pth', map_location='cuda:0')

# # 提取模型权重（关键！）
# model_weights = ckpt['state_dict']  # 或 ckpt['model']，取决于 NanoDet 版本
# print('keys:',model_weights.keys())

# 保存纯权重（无优化器）
# torch.save(model_weights, 'nanodet-plus-m_416_Feng_model_only.pth')

#去除仅训练时用到的aux开头的权重

import torch
import argparse

def remove_aux_keys(input_pth, output_pth):
    # 加载原始 checkpoint
    ckpt = torch.load(input_pth, map_location='cpu')
    
    # 检查是否包含 'state_dict' 或直接是 state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        is_checkpoint = True
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        is_checkpoint = False
    else:
        raise ValueError("Unsupported .pth format")

    # 过滤掉以 'aux.' 开头的 key
    new_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('aux_')
    }

    print(f"Original keys: {len(state_dict)}")
    print(f"Keys after removing 'aux_*': {len(new_state_dict)}")
    print("Removed keys (sample):")
    removed = [k for k in state_dict if k.startswith('aux_')]
    for k in removed[:5]:  # 打印前5个被移除的key
        print(f"  - {k}")
    if len(removed) > 5:
        print(f"  ... and {len(removed) - 5} more.")

    # 构建新的 checkpoint（保持原结构）
    if is_checkpoint:
        new_ckpt = ckpt.copy()
        new_ckpt['state_dict'] = new_state_dict
    else:
        new_ckpt = new_state_dict

    # 保存新文件
    torch.save(new_ckpt, output_pth)
    print(f"\nSaved cleaned model to: {output_pth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'aux.' keys from .pth model")
    parser.add_argument("--input", default='workspace/nanodet-plus-m_416_Feng/model_best/nanodet_model_best.pth',help="Input .pth file path")
    parser.add_argument("--output", default='model_best_new.pth',help="Output .pth file path")
    args = parser.parse_args()

    remove_aux_keys(args.input, args.output)