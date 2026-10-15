# gqa_compare.py
# 实验 8 对比：标准 MHA（实验5 RoPE 版）vs GQA（K/V 头共享）
# 同一词表（65）、同 seed、同训练配置，唯一变量 = n_kv_head（4 → 2）
# 关注点：省了多少参数 / loss 有没有掉 / 谁学得更快
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_csv(path):
    steps, trains, vals = [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            s, tr, va, _ = line.strip().split(",")
            steps.append(int(s)); trains.append(float(tr)); vals.append(float(va))
    return steps, trains, vals


def main():
    mha = read_csv("loss_log_rope.csv")   # MHA：807,936 参数（实验5 RoPE 最优版）
    gqa = read_csv("loss_log_gqa.csv")    # GQA：742,400 参数（实验8，n_kv_head=2）

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(mha[0], mha[1], color="#9EACEA", linewidth=1.8, label="MHA train（807,936 参数）")
    ax.plot(mha[0], mha[2], color="#9EACEA", linewidth=1.8, linestyle="--", label="MHA val")
    ax.plot(gqa[0], gqa[1], color="#52C41A", linewidth=1.8, label="GQA train（742,400 参数）")
    ax.plot(gqa[0], gqa[2], color="#52C41A", linewidth=1.8, linestyle="--", label="GQA val")
    ax.set_title("实验 8：GQA 分组查询注意力 vs 标准 MHA（省 65,536 参数 ≈ 8.1%）", fontsize=13)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/loss_gqa_compare.png", dpi=150)
    print("对比图已保存到 assets/loss_gqa_compare.png")

    # 顺便打印训练结束时双方的 val loss，方便写 README
    print(f"MHA 最终: train {mha[1][-1]:.4f} | val {mha[2][-1]:.4f}")
    print(f"GQA 最终: train {gqa[1][-1]:.4f} | val {gqa[2][-1]:.4f}")


if __name__ == "__main__":
    main()
