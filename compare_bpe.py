# compare_bpe.py
# 实验 6 对比：字符级 vs BPE 级模型，同 seed 同温度生成文本
# 同时画 loss 曲线对比图（两个子图，各自带随机基线）
import argparse
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows 自带微软雅黑，避免中文标题显示成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from model import GPT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_model(path):
    """加载模型 + 分词器，返回 (model, encode, decode)"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
    if ckpt.get("tokenizer_type", "char") == "bpe":
        from tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.merges = ckpt["merges"]
        tok.vocab = ckpt["vocab"]
        tok.char_to_id = ckpt["char_to_id"]
        return model, tok.encode, tok.decode
    else:
        stoi, itos = ckpt["stoi"], ckpt["itos"]
        return model, (lambda s: [stoi[c] for c in s]), (lambda ids: "".join(itos[int(i)] for i in ids))


def main():
    # ---------- 生成对比 ----------
    torch.manual_seed(1337)  # 同 seed 保证公平
    models = [
        ("字符级（实验5最优）", "model_rope.pt"),
        ("BPE 级（词表2065）", "model_bpe.pt"),
    ]
    for name, path in models:
        model, encode, decode = load_model(path)
        prompt = "ROMEO:"
        context = torch.tensor([encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(context, 300, temperature=0.8)
        print(f"\n{'='*60}\n{name}（{path}）\n{'='*60}")
        print(decode(out[0].tolist())[:1000])

    # ---------- loss 曲线对比图 ----------
    def read_csv(path):
        steps, trains, vals = [], [], []
        with open(path, encoding="utf-8") as f:
            next(f)  # 跳过表头
            for line in f:
                s, tr, va, _ = line.strip().split(",")
                steps.append(int(s)); trains.append(float(tr)); vals.append(float(va))
        return steps, trains, vals

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    titles = [
        ("字符级（词表 65）", "loss_log.csv", 4.174, "ln(65) ≈ 4.17"),
        ("BPE 级（词表 2065）", "loss_log_bpe.csv", 7.633, "ln(2065) ≈ 7.63"),
    ]
    for ax, (name, log, base, base_label) in zip(axes, titles):
        steps, trains, vals = read_csv(log)
        ax.plot(steps, trains, label="train", color="#9EACEA", linewidth=2)
        ax.plot(steps, vals, label="val", color="#EA6668", linewidth=2)
        ax.axhline(base, color="#6B7280", linestyle="--", linewidth=1)
        ax.text(steps[0] + 30, base + 0.15, f"随机基线 {base_label}", color="#6B7280", fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("训练步数")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("实验 6：字符级 vs BPE —— loss 不能直接跨词表比较（随机基线不同）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("assets/loss_bpe_compare.png", dpi=150)
    print("\n对比图已保存到 assets/loss_bpe_compare.png")


if __name__ == "__main__":
    main()
