# -*- coding: utf-8 -*-
"""
================================================================================
before_after_demo.py —— 训练前后生成效果对比（实验 3）
================================================================================

【这个文件是干嘛的】
    同一句话开头、同一种子、同一温度下，对比两个模型生成的文本：
        模型 A：随机初始化（完全没训练）—— 应该输出乱码
        模型 B：训练 2000 步后的 model.pt —— 应该输出莎士比亚腔

【怎么运行】
    python before_after_demo.py

【看什么】
    训练前：字符完全随机拼凑，因为权重是随机数，模型"什么都不知道"
    训练后：有角色名、有分行、句子半通不通 —— 权重里存了"谁和谁经常挨着"
"""

import os
import sys

import torch

from model import GPT

# Windows 控制台中文输出兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def generate_text(model, prompt, length, temperature, seed):
    """用给定模型生成一段文本（固定种子，保证可复现）"""
    torch.manual_seed(seed)  # 固定随机种子，两次生成可对比
    itos = model.itos  # id → 字符

    encode = lambda s: [model.stoi[c] for c in s]
    context = torch.tensor([encode(prompt)], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model.generate(context, length, temperature=temperature)

    return "".join(itos[int(i)] for i in out[0].tolist())


def main():
    # ---- 参数 ----
    prompt = "ROMEO:"          # 开头（和 README 示例一致）
    length = 300               # 生成多少字符
    temperature = 0.8          # 中等随机性
    seed = 1337                # 固定种子（和训练一致）

    # ---- 读训练好的模型配置 ----
    ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    def build_model(load_weights):
        """按训练时的配置建模型；load_weights=True 时加载训练好的权重"""
        m = GPT(
            ckpt["vocab_size"],
            n_embd=cfg["n_embd"],
            n_head=cfg["n_head"],
            n_layer=cfg["n_layer"],
            block_size=cfg["block_size"],
            activation=cfg.get("activation", "gelu"),
        )
        if load_weights:
            m.load_state_dict(ckpt["model"])  # 把训练好的权重"装"进去
        # 把词表映射挂到模型上（generate_text 用）
        m.stoi = ckpt["stoi"]
        m.itos = ckpt["itos"]
        return m

    # ---- 模型 A：随机初始化（没训练） ----
    model_rand = build_model(load_weights=False)
    text_before = generate_text(model_rand, prompt, length, temperature, seed)

    # ---- 模型 B：训练好的 ----
    model_trained = build_model(load_weights=True)
    text_after = generate_text(model_trained, prompt, length, temperature, seed)

    # ---- 输出 ----
    os.makedirs("assets", exist_ok=True)

    print("=" * 60)
    print("【训练前：随机初始化模型】（权重是随机数，什么都不知道）")
    print("=" * 60)
    print(text_before)
    print()
    print("=" * 60)
    print("【训练后：2000 步训练好的模型】")
    print("=" * 60)
    print(text_after)

    # 保存到文件，方便 README 引用
    with open("assets/before_train.txt", "w", encoding="utf-8") as f:
        f.write(text_before)
    with open("assets/after_train.txt", "w", encoding="utf-8") as f:
        f.write(text_after)
    print("\n已保存到 assets/before_train.txt 和 assets/after_train.txt")


if __name__ == "__main__":
    main()
