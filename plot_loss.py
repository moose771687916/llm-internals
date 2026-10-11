# -*- coding: utf-8 -*-
"""
================================================================================
plot_loss.py —— 把训练日志画成 loss 曲线图（README 首图 / 消融实验对比图）
================================================================================

【这个文件是干嘛的】
    读取 train.py --log 参数生成的 CSV 日志，画出 loss 随训练步数的下降曲线。
    也支持传入第二份日志（--log2），画两个版本模型的对比图（消融实验用）。

【怎么用】
    1. 训练并记录日志：
       python train.py --log loss_log.csv --eval_interval 50
       （--eval_interval 50 表示每 50 步记录一次，曲线更平滑）
    2. 画单条曲线：
       python plot_loss.py --log loss_log.csv --out assets/loss_curve.png
    3. 画两条曲线对比（比如 GELU vs SwiGLU 消融实验）：
       python plot_loss.py --log loss_log_gelu.csv --log2 loss_log_swiglu.csv \
           --out assets/loss_compare.png --name1 "GELU" --name2 "SwiGLU"

【图里看什么】
    - train loss（实线）：模型在训练数据上的误差，应该一路下降
    - val loss（虚线）：在没见过的验证数据上的误差，检验有没有"死记硬背"
    - 两条线都在降 → 模型真的在学规律，不是背答案
    - 对比模式下：谁的曲线更低，谁的"表达能力"更强
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # 无界面模式：不弹窗，直接保存图片文件
import matplotlib.pyplot as plt

# Windows 中文字体设置（防止图里中文变成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def load_log(path):
    """读取一份训练日志 CSV，返回 (steps, train_losses, val_losses)"""
    steps, train_losses, val_losses = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):  # DictReader 按表头自动解析每一行
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return steps, train_losses, val_losses


def plot_single(ax, steps, trains, vals, title, out):
    """画单条日志的 train/val 曲线"""
    ax.plot(steps, trains, label="train loss（训练集）", color="#4C78A8",
            linewidth=2, marker="o", markersize=4)
    ax.plot(steps, vals, label="val loss（验证集）", color="#F58518",
            linewidth=2, marker="s", markersize=4)

    # 标注起点（随机瞎猜的水平 ln(65)≈4.17）和终点
    ax.axhline(4.17, color="#9E9E9E", linestyle="--", linewidth=1)
    ax.text(steps[0], 4.22, "随机瞎猜 ≈ 4.17 (ln 65)", fontsize=9, color="#757575")

    ax.set_xlabel("训练步数 step")
    ax.set_ylabel("交叉熵损失 loss（越低越好）")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.annotate(
        f"最终 train {trains[-1]:.3f}",
        xy=(steps[-1], trains[-1]),
        xytext=(steps[-1] * 0.72, trains[-1] + 0.25),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#4C78A8"),
    )
    ax.annotate(
        f"最终 val {vals[-1]:.3f}",
        xy=(steps[-1], vals[-1]),
        xytext=(steps[-1] * 0.55, vals[-1] + 0.25),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#F58518"),
    )


def plot_compare(ax, data1, data2, name1, name2, title, out):
    """对比两份日志：各画 train/val 两条线（实线=训练集，虚线=验证集）"""
    colors = {"1": "#4C78A8", "2": "#E45756"}
    for (steps, trains, vals), key, name in [
        (data1, "1", name1), (data2, "2", name2)]:
        ax.plot(steps, trains, label=f"{name} train", color=colors[key],
                linewidth=2, marker="o", markersize=3)
        ax.plot(steps, vals, label=f"{name} val", color=colors[key],
                linewidth=1.4, linestyle="--", marker="s", markersize=3)

    ax.axhline(4.17, color="#9E9E9E", linestyle="--", linewidth=1)
    ax.text(0, 4.22, "随机瞎猜 ≈ 4.17", fontsize=9, color="#757575")

    ax.set_xlabel("训练步数 step")
    ax.set_ylabel("交叉熵损失 loss（越低越好）")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 标注两个版本的最终值
    for (steps, trains, vals), color, name in [
        (data1, "#4C78A8", name1), (data2, "#E45756", name2)]:
        ax.annotate(
            f"{name} 最终 train {trains[-1]:.3f} / val {vals[-1]:.3f}",
            xy=(steps[-1], trains[-1]),
            xytext=(steps[-1] * 0.6, trains[-1] + 0.2),
            fontsize=9.5,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color),
        )


def main():
    ap = argparse.ArgumentParser(description="把训练日志画成 loss 曲线")
    ap.add_argument("--log", default="loss_log.csv", help="训练日志 CSV 路径")
    ap.add_argument("--log2", default="", help="第二份日志 CSV 路径（可选，画对比图）")
    ap.add_argument("--name1", default="GELU", help="第一份日志的名称（对比图图例用）")
    ap.add_argument("--name2", default="SwiGLU", help="第二份日志的名称（对比图图例用）")
    ap.add_argument("--out", default="assets/loss_curve.png", help="输出图片路径")
    args = ap.parse_args()

    # ---- 读取日志 ----
    steps1, trains1, vals1 = load_log(args.log)
    if not steps1:
        print("日志为空，先运行：python train.py --log loss_log.csv")
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    if args.log2:
        steps2, trains2, vals2 = load_log(args.log2)
        if not steps2:
            print("第二份日志为空：", args.log2)
            return
        plot_compare(ax, (steps1, trains1, vals1), (steps2, trains2, vals2),
                     args.name1, args.name2,
                     f"{args.name1} vs {args.name2} 消融对比", args.out)
        n = len(steps1) + len(steps2)
        msg = (f"对比图已保存到 {args.out}\n"
               f"{args.name1}: {len(steps1)} 个点，train {trains1[0]:.3f}→{trains1[-1]:.3f}，"
               f"val {vals1[0]:.3f}→{vals1[-1]:.3f}\n"
               f"{args.name2}: {len(steps2)} 个点，train {trains2[0]:.3f}→{trains2[-1]:.3f}，"
               f"val {vals2[0]:.3f}→{vals2[-1]:.3f}")
    else:
        plot_single(ax, steps1, trains1, vals1,
                    "Metron 迷你 GPT 训练曲线（824,320 参数 · 2000 步）", args.out)
        n = len(steps1)
        msg = (f"曲线图已保存到 {args.out}\n"
               f"共记录 {len(steps1)} 个点：step {steps1[0]} → {steps1[-1]}，"
               f"train {trains1[0]:.3f} → {trains1[-1]:.3f}，"
               f"val {vals1[0]:.3f} → {vals1[-1]:.3f}")

    fig.tight_layout()

    # ---- 保存 ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(msg)


if __name__ == "__main__":
    main()
