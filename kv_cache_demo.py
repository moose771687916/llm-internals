# -*- coding: utf-8 -*-
"""
================================================================================
kv_cache_demo.py —— 实验 7：KV Cache 推理加速对比
================================================================================

【这个文件是干嘛的】
    同一个模型、同一个 prompt、同一个随机种子，分别用两种方式生成同样长度的文本：
      1. 朴素版（无缓存）：model.generate()      —— 每步把整个序列重新过一遍模型
      2. KV Cache 版：model.generate_cached()    —— 预填充后，每步只算最后 1 个位置
    然后对比：
      - 生成结果是否逐字一致（正确性验证：缓存不能改变输出）
      - 生成耗时和加速比（性能验证：缓存省了多少重复计算）

【怎么运行】
    python kv_cache_demo.py
    python kv_cache_demo.py --model model_rope.pt --length 120
    python kv_cache_demo.py --prompt "KING:" --temperature 0.5
"""

# ==============================================================================
# 【import 部分】
# ==============================================================================
import sys
import time  # 计时用（time.perf_counter 精度高）

import argparse
import torch

from model import GPT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="KV Cache 推理加速对比（实验 7）")
    ap.add_argument("--model", default="model_bpe.pt", help="权重文件（建议用 RoPE 版，效果最好）")
    ap.add_argument("--prompt", default="ROMEO:", help="开头文字")
    ap.add_argument("--length", type=int, default=100, help="生成多少个 token")
    ap.add_argument("--temperature", type=float, default=0.8, help="随机性")
    args = ap.parse_args()

    # ==========================================================================
    # 【第 1 步：加载模型】（和 generate.py 完全一样的加载逻辑）
    # ==========================================================================
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(
        ckpt["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        activation=cfg.get("activation", "gelu"),
        pos_enc=cfg.get("pos_enc", "abs"),
        rope_base=cfg.get("rope_base", 10000.0),
        n_kv_head=cfg.get("n_kv_head", None),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer_type = ckpt.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        from tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.merges = ckpt["merges"]
        tok.vocab = ckpt["vocab"]
        tok.char_to_id = ckpt["char_to_id"]
        encode, decode = tok.encode, tok.decode
    else:
        itos, stoi = ckpt["itos"], ckpt["stoi"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[int(i)] for i in ids)

    context = torch.tensor([encode(args.prompt)], dtype=torch.long)
    prefill_len = context.shape[1]
    assert prefill_len + args.length <= cfg["block_size"], \
        f"预填充 {prefill_len} + 生成 {args.length} 超过 block_size={cfg['block_size']}（cos/sin 表按它预计算）"

    print(f"模型：{args.model}（{tokenizer_type} 分词，{ckpt['vocab_size']} 词表，"
          f"{cfg['n_layer']} 层 {cfg['n_embd']} 维，位置编码={cfg.get('pos_enc','abs')}）")
    print(f"prompt：{args.prompt!r}（{prefill_len} token），生成 {args.length} token，温度 {args.temperature}")
    print()

    # ==========================================================================
    # 【第 2 步：朴素版（无 KV Cache）】
    # ==========================================================================
    torch.manual_seed(1337)          # 固定随机种子，保证两次生成"从同样的随机起点出发"
    t0 = time.perf_counter()
    with torch.no_grad():
        out_naive = model.generate(context, args.length, temperature=args.temperature)
    t_naive = time.perf_counter() - t0
    text_naive = decode(out_naive[0].tolist())

    # ==========================================================================
    # 【第 3 步：KV Cache 版】
    # ==========================================================================
    torch.manual_seed(1337)          # 同样的种子（随机抽样序列会一模一样）
    t0 = time.perf_counter()
    with torch.no_grad():
        out_cached = model.generate_cached(context, args.length, temperature=args.temperature)
    t_cached = time.perf_counter() - t0
    text_cached = decode(out_cached[0].tolist())

    # ==========================================================================
    # 【第 4 步：对比验证】
    # ==========================================================================
    # 正确性：两个版本必须在"同样的随机序列"下生成逐字相同的文本
    # （KV Cache 只是省重复计算，不能改变任何数学结果——这是它的黄金检验标准）
    same = (out_naive == out_cached).all().item()
    speedup = t_naive / t_cached if t_cached > 0 else float("inf")

    print("=" * 70)
    print(f"结果逐字一致：{'✅ 是' if same else '❌ 否（有 bug！）'}")
    print("=" * 70)
    print(f"朴素版（无缓存）：  {t_naive * 1000:8.1f} ms   （每步重算全部历史位置）")
    print(f"KV Cache 版：      {t_cached * 1000:8.1f} ms   （每步只算最后 1 个位置）")
    print(f"加速比：           {speedup:8.2f} ×")
    print(f"每秒生成 token：   朴素 {args.length / t_naive:6.0f}  vs  Cache {args.length / t_cached:6.0f}")
    print()
    print("生成文本（Cache 版）：")
    print("-" * 70)
    print(text_cached)
    print("-" * 70)
    if not same:
        print("⚠️ 两版结果不一致——KV Cache 实现有 bug，请检查 pos_offset / cache 拼接逻辑！")


if __name__ == "__main__":
    main()
