# -*- coding: utf-8 -*-
"""
================================================================================
generate.py —— 用训练好的模型生成文本
================================================================================

【这个文件是干嘛的】
    读取 train.py 训练好的权重文件（model.pt），然后给定一个开头（prompt），
    让模型一个字一个字地"预测下一个字符"，循环生成一段文本。

【生成的本质】
    模型不会"一次性写出整段话"，它是这样工作的：
        输入 "ROMEO:" → 模型预测下一个字符最可能是 "\n"（换行）
        输入 "ROMEO:\n" → 预测下一个最可能是 "B"
        输入 "ROMEO:\nB" → 预测下一个最可能是 "u"
        ... 重复 500 次，就生成了 500 个字符
    每一步都只看"前面的字符"，预测"下一个字符"。这就是大模型生成文本的方式。

【怎么运行】
    python generate.py                                    # 默认从 "ROMEO:" 生成 500 字
    python generate.py --prompt "KING:" --length 300     # 自定义开头和长度
    python generate.py --temperature 0.3                  # 更保守（输出稳定但死板）
    python generate.py --temperature 1.5                  # 更随机（有惊喜但可能乱）
"""

# ==============================================================================
# 【import 部分】
# ==============================================================================
import sys
# ↑ 系统库，用来做 Windows 控制台中文编码设置（和 train.py 一样）

import argparse
# ↑ 命令行参数解析（和 train.py 一样）

import torch
# ↑ PyTorch 深度学习库

from model import GPT
# ↑ 从 model.py 导入 GPT 类（要先按图纸建一个空模型，再把训练好的权重"装"进去）

# Windows 控制台中文输出兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    # ==========================================================================
    # 【第 0 步：解析命令行参数】
    # ==========================================================================
    ap = argparse.ArgumentParser(description="用训练好的迷你 GPT 生成文本")
    ap.add_argument("--model", default="model.pt", help="训练好的权重文件路径")
    ap.add_argument("--prompt", default="ROMEO:", help="开头文字（模型从这里开始往下编）")
    ap.add_argument("--length", type=int, default=500, help="生成多少个字符")
    ap.add_argument("--temperature", type=float, default=0.8, help="随机性（0.1保守 ~ 1.5疯狂）")
    args = ap.parse_args()

    # ==========================================================================
    # 【第 1 步：加载训练好的模型】
    # ==========================================================================
    # torch.load 读取之前保存的二进制文件，还原成 Python 对象（一个字典）
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    # ↑ map_location="cpu"：把模型加载到 CPU 上（即使之前在 GPU 上训练的也能在 CPU 上用）
    #   weights_only=False：允许加载字典里的非张量对象（我们存了 stoi/itos 这些字典）
    #   （PyTorch 2.6+ 默认 weights_only=True，会拒绝加载普通 Python 对象，所以要显式设 False）

    cfg = ckpt["config"]
    # ↑ 取出训练时的超参数配置（n_embd、n_head、n_layer、block_size）
    #   必须用和训练时完全一样的结构建模型，否则权重对不上（形状不匹配会报错）

    # 按训练时的结构，建一个"空的" GPT 模型（权重还是随机的）
    model = GPT(
        ckpt["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        activation=cfg.get("activation", "gelu"),  # 兼容 SwiGLU 版模型（旧模型默认 gelu）
        pos_enc=cfg.get("pos_enc", "abs"),          # 兼容 RoPE 版模型（旧模型默认 abs）
        rope_base=cfg.get("rope_base", 10000.0),    # 兼容改过 base 的 RoPE 模型（默认 10000）
    )

    # 把训练好的权重"装"进这个空模型
    model.load_state_dict(ckpt["model"])
    # ↑ load_state_dict 把保存的权重字典赋值给模型各层。
    #   这一步之后，模型就从"随机权重"变成了"会写莎士比亚的模型"。

    model.eval()
    # ↑ 切换到"评估模式"。和训练模式的区别：
    #   - Dropout 层不再随机丢弃（推理时要稳定输出）
    #   - BatchNorm 等层用训练时统计的均值方差，而不是当前 batch 的
    #   我们的模型里有 Dropout，所以必须调 eval()，否则每次生成结果会随机乱跳。

    # 取出词表映射
    itos = ckpt["itos"]  # id → 字符，生成完用来把数字变回文字
    stoi = ckpt["stoi"]  # 字符 → id，用来把 prompt 变成数字

    # ==========================================================================
    # 【第 2 步：把开头（prompt）变成数字】
    # ==========================================================================
    encode = lambda s: [stoi[c] for c in s]
    # ↑ 和 train.py 里一样的编码函数：字符串 → id 列表

    context = torch.tensor([encode(args.prompt)], dtype=torch.long)
    # ↑ 把 prompt 编码成 id 列表，再包一层变成 (1, T) 形状的张量
    #   前面的 1 是 batch 大小（一次只生成一条）
    #   比如 prompt="ROMEO:" → [44, 50, 46, 38, 50, 27] → 形状 (1, 6)

    # ==========================================================================
    # 【第 3 步：生成文本】
    # ==========================================================================
    with torch.no_grad():
        # ↑ 推理时不需要计算梯度，用 no_grad 省内存、加速
        out = model.generate(context, args.length, temperature=args.temperature)
        # ↑ 调用 GPT 类里的 generate 方法（在 model.py 里有详细注释）
        #   它会循环 length 次，每次预测下一个字符并拼到末尾
        #   out 的形状：(1, T + length)，包含 prompt + 新生成的所有字符 id

    # ==========================================================================
    # 【第 4 步：把数字变回文字，打印出来】
    # ==========================================================================
    generated_text = "".join(itos[int(i)] for i in out[0].tolist())
    # ↑ out[0] 取出第一条（也是唯一一条）数据，形状 (T+length,)
    #   .tolist() 把张量转成 Python 列表
    #   遍历每个 id，用 itos 字典查回字符，再用 "".join() 拼成一个字符串
    print(generated_text)
    # ↑ 打印生成的完整文本（包含开头的 prompt）


if __name__ == "__main__":
    main()
