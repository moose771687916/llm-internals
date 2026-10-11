# -*- coding: utf-8 -*-
"""
透视脚本：把 c_attn 和 c_proj 内部的权重矩阵"挖出来"给你看
运行：python peek_weights.py
"""
import torch
from model import GPT

# 1. 造两个一模一样的模型：一个不加载权重（训练前的随机数），一个加载训练好的
m_before = GPT(vocab_size=65, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.0)  # 随机初始化
m_after  = GPT(vocab_size=65, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.0)
saved = torch.load("model.pt", map_location="cpu")  # 外层是个字典
sd = saved["model"]  # 里面 "model" 才是所有权重的字典
m_after.load_state_dict(sd)

# 访问路径：模型 -> 第1个Block -> 注意力 -> c_attn / c_proj
attn_b = m_before.blocks[0].attn
attn_a = m_after.blocks[0].attn

def show(name, before_w, after_w, rows=3, cols=5):
    print("=" * 60)
    print(f"{name} 的形状：{list(before_w.shape)}  （行 × 列）")
    print(f"{name} 的参数总数：{before_w.numel()}")
    print()
    print(f"前 {rows} 行 × 前 {cols} 列（训练前随机数）：")
    for r in range(rows):
        row = [f"{before_w[r][c].item():+.4f}" for c in range(cols)]
        print("   [" + "  ".join(row) + "  ...]")
    print()
    print(f"同一个格子，训练后变成（第 0 行前 {cols} 个）：")
    for r in range(rows):
        row = [f"{after_w[r][c].item():+.4f}" for c in range(cols)]
        print("   [" + "  ".join(row) + "  ...]")
    print()
    print(f"对比：第 0 行第 0 列  训练前={before_w[0][0].item():+.4f}  →  训练后={after_w[0][0].item():+.4f}")
    print()

show("c_attn.weight（复印机的配方表 W：384 行 × 128 列）", attn_b.c_attn.weight, attn_a.c_attn.weight)
show("c_proj.weight（汇总员的配方表 W：128 行 × 128 列）", attn_b.c_proj.weight, attn_a.c_proj.weight)

print("=" * 60)
print("结论：里面真有东西（矩阵），而且训练会把这些数字改掉！")
