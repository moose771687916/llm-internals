# -*- coding: utf-8 -*-
"""
================================================================================
rope_extrapolation.py —— RoPE 外推能力验证（第 2 级升级的"灵魂实验"）
================================================================================

【这个脚本在验证什么】
    我们训练模型时，上下文只有 128 个字符（block_size=128），
    也就是说模型训练时"见过的位置"只有 0~127。

    绝对位置表（GPT-2 经典）的问题：位置信息靠"查表"。
    如果喂给它第 200 个位置的字符，表里根本没有第 200 行 → 直接报错。
    → 绝对位置表"永远无法外推"到训练时没见过的新位置。

    RoPE 的问题：位置信息靠"旋转公式"（θ_i = 10000^(-2i/d)），
    任何位置的角度都能算出来 → 位置 200、2000、100 万都能算。
    → RoPE "天然能外推"到训练时没见过的新位置。

【怎么验证】
    1. 加载训练好的 RoPE 模型（block_size=128 训练的）
    2. 用 block_size=256 重建模型（旋转表预计算到 256 个位置）
    3. 把一段 256 字符的真实莎士比亚文本喂给模型，算 loss
    4. 如果 loss 远低于随机基线（4.17，瞎猜的水平），说明
       模型能处理"训练时从未见过的新位置 128~255" → 外推成功！

【怎么运行】
    python rope_extrapolation.py                          # 默认验证 model_rope.pt
    python rope_extrapolation.py --model model_rope.pt    # 指定模型
"""

import sys
import argparse

import torch

from model import GPT

# Windows 控制台中文输出兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="验证 RoPE 位置编码的外推能力")
    ap.add_argument("--model", default="model_rope.pt", help="训练好的权重文件（必须是 rope 版）")
    ap.add_argument("--extrap_len", type=int, default=256,
                    help="外推到的序列长度（默认 256，超过训练时的 128）")
    ap.add_argument("--ntk", action="store_true",
                    help="启用 NTK-aware 缩放：只重算频率表（改 base），模型参数一个不动")
    args = ap.parse_args()

    # 随机基线：65 个字符等概率瞎猜，loss = log(65) ≈ 4.17
    random_baseline = 4.174

    # ---- 第 1 步：加载训练好的模型 ----
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"训练时上下文长度（block_size）: {cfg['block_size']}")
    print(f"训练时位置编码（pos_enc）: {cfg.get('pos_enc', 'abs')}")

    # ---- 第 2 步：用更大的 block_size 重建模型 ----
    # RoPE 的旋转表是按 block_size 预计算的：重建 256 就会预计算 256 个位置的 cos/sin。
    # 绝对位置表做不到（位置表只有 128 行，查不到 129+）。
    model = GPT(
        ckpt["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=args.extrap_len,   # ← 关键：外推到 256！
        activation=cfg.get("activation", "gelu"),
        pos_enc="rope",               # ← 必须 rope（绝对位置表无法外推）
        rope_base=cfg.get("rope_base", 10000.0),  # 用训练时的 base（Qwen 式改过 base 也能对上）
    )

    # 加载权重：过滤掉 buffer（mask/cos/sin 大小随 block_size 变，不匹配是正常的）
    sd = {k: v for k, v in ckpt["model"].items()
          if not (k.endswith(".mask") or k.endswith(".cos") or k.endswith(".sin"))}
    model.load_state_dict(sd, strict=False)
    # ↑ strict=False：允许 buffer 不匹配（mask/cos/sin 用新的大尺寸）
    #   其余参数（所有可训练权重）必须完整匹配

    # ---- 第 2.5 步：NTK-aware 缩放（--ntk 时）----
    # 核心思想：模型参数一个都不动，只把"转速公式里的 base"换大，
    # 让新位置（128~255）的旋转角度落回模型训练时熟悉的范围内。
    # 效果（社区标准公式）：高频钟几乎不动（保住近处精度），
    # 低频钟按 scale 压慢（撑开远处）——一条公式自动完成渐变。
    if args.ntk:
        head_dim = model.blocks[0].attn.head_dim
        scale = args.extrap_len / cfg["block_size"]  # 长度倍数（256/128=2）
        new_base = 10000 * (scale ** (head_dim / (head_dim - 2)))
        # ↑ NTK 标准公式：new_base = base × scale^(d/(d-2))
        #   d=32, scale=2 → 2^(32/30)≈2.09 → new_base ≈ 20,930
        print(f"NTK 缩放：scale={scale:.1f}，base 从 10,000 → {new_base:,.0f}")
        inv_freq = 1.0 / (new_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(args.extrap_len).float().unsqueeze(1) * inv_freq.unsqueeze(0)
        new_cos = t.cos().view(1, 1, args.extrap_len, -1)
        new_sin = t.sin().view(1, 1, args.extrap_len, -1)
        # 用新频率表替换每个注意力层的 cos/sin（模型参数一个没动）
        for block in model.blocks:
            block.attn.cos = new_cos
            block.attn.sin = new_sin

    model.eval()

    # ---- 第 3 步：准备一段超过 128 字符的真实莎士比亚文本 ----
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    # 用训练数据源重新下载一段文本（和 train.py 相同的数据源）
    import urllib.request
    url = "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
    data = urllib.request.urlopen(url).read().decode("utf-8")
    # 截取开头 256 个字符作为"长文本输入"
    text = data[:args.extrap_len]
    ids = [stoi[c] for c in text]  # 字符 → id
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, 256)

    # ---- 第 4 步：算模型在这个长文本上的 loss ----
    # 注意：训练时目标是"下一个字符"（X 的移位版本），不是 X 自己！
    # 所以这里输入前 L-1 个字符，目标是后 L-1 个字符（标准的下一个字符预测）
    x_in = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)  # (1, L-1)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)       # (1, L-1)
    with torch.no_grad():
        logits, loss = model(x_in, y)  # 输入前 L-1 个，目标后 L-1 个

    mode = "NTK 缩放外推" if args.ntk else "裸 RoPE 外推"
    print()
    print("=" * 60)
    print(f"随机基线（瞎猜）:            {random_baseline:.3f}")
    print(f"{mode} 256 字符长文本:       {loss.item():.3f}")
    print("=" * 60)
    if loss.item() < random_baseline - 1.0:
        print("结果：外推成功！模型能处理训练时从未见过的新位置（128~255）。")
        print("      这就是 RoPE 能让大模型支持 10 万、100 万上下文的技术基石。")
    else:
        print("结果：外推失败（loss 没有明显低于随机基线）。")
        print("      可能原因：训练步数太少 / 数据太少，模型还没学会足够规律。")

    # ---- 第 5 步：外推生成演示（可选）----
    # 给模型 128 个真实字符开头，让它继续写 100 个字符（全程用到 128~227 的新位置）
    print()
    print("--- 外推生成演示：从莎士比亚原文继续写 100 个字符（用到新位置）---")
    prompt_ids = ids[:128]  # 前 128 个字符（训练见过的位置范围）
    idx = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    gen = model.generate(idx, max_new_tokens=100, temperature=0.8)
    gen_text = "".join(itos[i] for i in gen[0].tolist())
    print(gen_text)


if __name__ == "__main__":
    main()
