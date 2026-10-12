# -*- coding: utf-8 -*-
"""
================================================================================
train.py —— 训练脚本（让模型从"什么都不会"变成"会写莎士比亚"）
================================================================================

【这个文件是干嘛的】
    它执行完整的训练流程：
        读数据 → 把文本变成数字 → 按 model.py 的图纸建模型 →
        反复喂数据、算误差、调权重 → 保存训练好的权重到 model.pt

【训练的本质（最重要的一句话）】
    模型一开始权重是随机数，什么都不会。
    每次喂一批数据 → 模型做出预测 → 和正确答案比误差(loss) →
    反向传播算出每个权重该往哪个方向调 → 优化器微调权重。
    反复几千几万次后，权重自动收敛成"一套能写出像样文字的评分公式"。
    整个过程没有任何人"教"它语法，它是从数据里自己学出来的。

【怎么运行】
    python train.py                              # 默认训练 2000 步
    python train.py --max_iters 5000             # 训练更多步
    python train.py --print_weights               # 训练完打印每层权重信息
    python train.py --n_layer 6 --n_embd 256     # 建一个更大的模型
"""

# ==============================================================================
# 【import 部分】
# ==============================================================================
import sys
# ↑ sys 是 Python 标准库，提供和"系统/解释器"交互的功能。
#   我们用它来做 Windows 控制台的中文编码设置（下面会看到）。

import time
# ↑ time 是 Python 标准库，提供时间相关功能。
#   我们用 time.time() 来计时，看训练每一步花了多久。

import argparse
# ↑ argparse 是 Python 标准库，用来解析"命令行参数"。
#   就是你在命令行输入的 --max_iters 2000 这种东西，argparse 帮你自动解析成变量。
#   有了它，不用改代码就能调整训练参数。

import math
# ↑ math 是 Python 标准库，提供数学函数。
#   我们用 math.cos（余弦）来实现"余弦退火"学习率调度（后面会看到）。

import torch
# ↑ 导入 PyTorch（深度学习库），和 model.py 里一样。

from model import GPT
# ↑ 从同目录下的 model.py 文件里导入 GPT 这个类。
#   这样我们就能在这里"按图纸造模型"了。
#   注意：from model import GPT 要求 train.py 和 model.py 在同一个文件夹里。

# Windows 控制台中文输出兼容（防止中文乱码）
if sys.platform == "win32":
    # ↑ sys.platform 返回当前操作系统，Windows 上是 "win32"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # ↑ 把标准输出的编码改成 UTF-8。Windows 控制台默认可能是 GBK，
    #   打印中文时可能乱码，改成 UTF-8 就稳了。errors="replace" 表示
    #   遇到无法编码的字符时用 ? 代替，而不是直接报错崩溃。


def main():
    # ==========================================================================
    # 【第 0 步：解析命令行参数】
    # ==========================================================================
    # argparse 的用法：先创建一个解析器，然后一个个"注册"参数，最后 parse_args() 拿到结果。
    ap = argparse.ArgumentParser(description="从零训练一个迷你 GPT")
    # ↑ 创建参数解析器，description 是 --help 时显示的描述。

    # 注册每个参数。add_argument 的参数：
    #   --data：参数名（命令行里用 --data xxx）
    #   default：不写时的默认值
    #   type：期望的类型（int/float/str），argparse 会自动转换
    #   help：--help 时显示的帮助文字
    #   action="store_true"：表示这是一个"开关"，写了就是 True，不写就是 False
    ap.add_argument("--data", default="input.txt", help="训练文本路径")
    ap.add_argument("--max_iters", type=int, default=2000, help="训练步数")
    ap.add_argument("--eval_interval", type=int, default=200, help="每多少步打印一次 loss")
    ap.add_argument("--batch_size", type=int, default=64, help="每批几条样本")
    ap.add_argument("--block_size", type=int, default=128, help="上下文长度（每条样本看多少个字符）")
    ap.add_argument("--n_layer", type=int, default=4, help="Transformer 层数（权重层数）")
    ap.add_argument("--n_head", type=int, default=4, help="注意力头数")
    ap.add_argument("--n_embd", type=int, default=128, help="向量维度")
    ap.add_argument("--lr", type=float, default=3e-4, help="学习率（每次更新权重的步长）")
    ap.add_argument("--warmup_iters", type=int, default=100,
                    help="学习率热身步数（前这么多步从 0 线性升到 lr，之后余弦退火）")
    ap.add_argument("--lr_min", type=float, default=None,
                    help="学习率退火最低值（默认 = 峰值 lr 的 1/10，不降到 0）")
    ap.add_argument("--print_weights", action="store_true", help="训练结束后打印每层权重信息")
    ap.add_argument("--save", default="model.pt", help="保存的模型文件名")
    ap.add_argument("--log", default="", help="训练日志 CSV 路径（记录每步 loss，留空则不记录）")
    ap.add_argument("--activation", default="gelu", choices=["gelu", "swiglu"],
                    help="MLP 激活函数：gelu（经典）或 swiglu（现代大模型版）")
    ap.add_argument("--pos_enc", default="abs", choices=["abs", "rope"],
                    help="位置编码：abs（绝对位置表，GPT-2 经典）或 rope（旋转位置编码，现代大模型）")
    ap.add_argument("--rope_base", type=float, default=10000.0,
                    help="RoPE 转速公式的 base（默认 10000；Qwen 训练时调到百万级以支持长上下文）")

    args = ap.parse_args()
    # ↑ 解析命令行，结果存在 args 里。比如 args.max_iters 就是你输入的训练步数。

    torch.manual_seed(1337)
    # ↑ 设置随机种子，保证每次运行结果一致（和 model.py 里一样的道理）。

    # ==========================================================================
    # 【第 1 步：读数据 + 字符级分词（Tokenizer）】
    # ==========================================================================
    # 神经网络只能处理数字，不能直接处理文字。所以要先把文本变成数字。
    # 我们用最简单的"字符级分词"：每个字符对应一个数字 id。
    # （大模型用的是更复杂的 BPE 分词，把常见字组合并成一个 token，但原理一样。）

    text = open(args.data, encoding="utf-8").read()
    # ↑ 打开训练文本文件，用 UTF-8 编码读取全部内容，存在 text 字符串里。
    #   莎士比亚数据大约 110 万个字符。

    chars = sorted(list(set(text)))
    # ↑ set(text) 找出文本里所有"不重复"的字符（比如 a、b、逗号、换行等）
    #   list() 转成列表，sorted() 排序（保证每次顺序一致）
    #   莎士比亚数据里有 65 种不同字符
    vocab_size = len(chars)
    # ↑ 词表大小 = 有多少种不同字符

    # 建立两个映射字典：
    stoi = {ch: i for i, ch in enumerate(chars)}
    # ↑ stoi = string to int，字符 → id。比如 {'a': 0, 'b': 1, ...}
    #   这是一个"字典推导式"，遍历 chars 列表，每个字符分配一个递增的 id。
    itos = {i: ch for i, ch in enumerate(chars)}
    # ↑ itos = int to string，id → 字符。比如 {0: 'a', 1: 'b', ...}
    #   生成完文本后，用这个字典把数字变回字符。

    encode = lambda s: [stoi[c] for c in s]
    # ↑ encode 是一个小函数（lambda 匿名函数）：输入字符串，输出 id 列表。
    #   比如 encode("abc") → [0, 1, 2]
    #   这就是"分词"：把文字变成模型能处理的数字。

    data = torch.tensor(encode(text), dtype=torch.long)
    # ↑ 把整个文本编码成 id 列表，再转成 PyTorch 张量（long 类型 = 64位整数）
    #   data 的形状：(1115394,)，一个一维数组，每个元素是一个字符的 id

    print(f"数据大小: {len(data):,} 字符 | 词表大小: {vocab_size}")
    # ↑ f-string 格式化输出。{len(data):,} 表示数字加千分位逗号（1,115,394）

    # ---- 划分训练集和验证集 ----
    n = int(0.9 * len(data))
    train_data = data[:n]   # 前 90% 当训练集（用来学权重）
    val_data = data[n:]     # 后 10% 当验证集（用来检查有没有"死记硬背"）
    # ↑ 为什么要分验证集：如果模型只在训练集上 loss 低、验证集上 loss 高，
    #   说明它在"背答案"（过拟合），没有真正学会规律。验证集就是"监考老师"。

    # ==========================================================================
    # 【第 2 步：建模型】
    # ==========================================================================
    model = GPT(
        vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        activation=args.activation,  # 激活函数：gelu 或 swiglu（消融实验用）
        pos_enc=args.pos_enc,        # 位置编码：abs（绝对位置表）或 rope（旋转位置编码）
        rope_base=args.rope_base,    # RoPE 转速公式的 base（Qwen 式：训练时调大支持长上下文）
    )
    # ↑ 按 model.py 里的图纸，创建一个 GPT 模型实例。
    #   这一步会：创建所有权重张量（随机初始化）、注册所有子模块。
    #   此时模型什么都不会，权重全是随机数。

    print(f"模型参数量: {model.count_parameters():,} （这就是「权重数字」的总个数）")

    # ==========================================================================
    # 【第 3 步：创建优化器】
    # ==========================================================================
    # 优化器是"更新权重的算法"。它知道每个权重该往哪个方向调、调多少。
    # AdamW 是目前最常用的优化器（GPT 也用它），是 Adam 的改进版（加了权重衰减）。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # ↑ model.parameters() 返回模型里所有可训练参数（权重）的迭代器
    #   lr=learning rate 学习率：每次更新权重的"步长"。
    #      太大会震荡（loss 下不去），太小会学得慢。3e-4 是 GPT 的标准值。

    # ---- 学习率调度器（第 1 级升级：让学习率"会自己变"）----
    # 固定学习率的问题是：前期大步子好（快点赶到谷底附近），
    # 但到了谷底附近还迈大步就会"来回震荡"，降不下去。
    # 所以业界做法是：前 warmup_iters 步从 0 线性升到 lr（热身，防起步震荡），
    # 之后按"余弦曲线"从 lr 慢慢降到 lr_min（默认 lr 的 1/10），精细收敛。
    # 注意：退火不是降到 0！降到 0 会让后期完全停止学习（实验 4a 验证过）。
    lr_min = args.lr_min if args.lr_min is not None else args.lr / 10
    # ↑ 退火的最低学习率。默认是峰值的 1/10（比如 3e-4 → 3e-5）。

    def lr_lambda(step):
        # ↑ 这个函数告诉调度器："第 step 步时，学习率应该是峰值的多少倍"
        if step < args.warmup_iters:
            # 热身阶段：从 0 线性升到 1（峰值的 1 倍 = 就是 lr）
            return (step + 1) / args.warmup_iters
        # 退火阶段：progress 从 0 走到 1，余弦从 1 平滑降到 lr_min/lr（默认 0.1）
        progress = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        return lr_min / args.lr + (1 - lr_min / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # ↑ PyTorch 的"学习率调度器"：每次 optimizer.step() 更新权重后，
    #   调用 scheduler.step() 就会按 lr_lambda 自动调整下一步的学习率。

    # ==========================================================================
    # 【第 4 步：定义"取一批数据"的函数】
    # ==========================================================================
    def get_batch(split):
        """
        从数据里随机取一批样本。
        split：'train' 或 'val'，决定从训练集还是验证集取
        返回：(x, y)
              x：输入，形状 (batch_size, block_size)，每条是一段连续字符
              y：目标，形状 (batch_size, block_size)，每条是 x 往后移一位的字符
                 （模型的任务：给定 x，预测 y）
        """
        d = train_data if split == "train" else val_data

        # 随机生成 batch_size 个起始位置（不能太靠后，否则取不够 block_size 个字符）
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))

        # 对每个起始位置，取 block_size 个连续字符作为输入 x
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        # 目标 y 是 x 往后移一位（预测下一个字符）
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])

        return x, y

    # ==========================================================================
    # 【第 5 步：训练循环（核心中的核心）】
    # ==========================================================================
    # ---- 训练日志（用于画 loss 曲线图）----
    log_file = None
    if args.log:
        log_file = open(args.log, "w", encoding="utf-8")
        log_file.write("step,train_loss,val_loss,time_s\n")
        # ↑ CSV 表头：步数、训练集 loss、验证集 loss、累计用时（秒）

    print("开始训练...（看到 loss 越来越小就说明模型在学东西）")
    t0 = time.time()  # 记录开始时间

    for step in range(args.max_iters):
        # ---- 5.1 取一批训练数据 ----
        xb, yb = get_batch("train")
        # xb：输入 (batch_size, block_size)
        # yb：正确答案 (batch_size, block_size)

        # ---- 5.2 前向传播：模型做出预测，算误差 ----
        logits, loss = model(xb, yb)
        # 调用 model(xb, yb) 会自动执行 GPT.forward()
        # logits：每个位置预测下一个字符的分数 (batch_size, block_size, vocab_size)
        # loss：预测和正确答案的差距，一个数。越小越好。

        # ---- 5.3 清零上一步的梯度 ----
        optimizer.zero_grad(set_to_none=True)
        # ↑ PyTorch 默认会"累加"梯度，所以每步开始前要清零。
        #   set_to_none=True 是更高效的清零方式（直接设成 None 而不是 0）。

        # ---- 5.4 反向传播：算出每个权重该调多少 ----
        loss.backward()
        # ↑ 这是 PyTorch 最神奇的功能：自动求导。
        #   它从 loss 出发，沿着计算图"反着走"，用链式法则算出
        #   每个权重对 loss 的"梯度"（权重往哪个方向调、调多少能让 loss 下降）。
        #   你不需要手写任何求导公式，PyTorch 全帮你算了。
        #   这就是"深度学习框架"的核心价值。

        # ---- 5.5 更新权重 ----
        optimizer.step()
        # ↑ 优化器根据刚才算出的梯度，按学习率更新所有权重。
        #   新权重 = 旧权重 - 学习率 × 梯度
        #   这一步之后，模型就"稍微聪明了一点点"。

        # ---- 5.5b 更新学习率（第 1 级升级）----
        scheduler.step()
        # ↑ 让学习率按 lr_lambda 的曲线走：热身期上升，之后余弦退火下降。
        #   当前学习率 = lr × lr_lambda(step)，可以从 optimizer 里读出来看。

        # ---- 5.6 定期打印进度 ----
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            # 用验证集算一下 loss（检查有没有过拟合）
            with torch.no_grad():
                # ↑ no_grad 表示这段代码不计算梯度（省内存，因为验证不需要反向传播）
                _, val_loss = model(*get_batch("val"))
                # ↑ * 是解包，把 get_batch 返回的 (x, y) 拆开传进去

            dt = time.time() - t0  # 累计用时
            cur_lr = optimizer.param_groups[0]["lr"]  # 当前学习率（调度器调出来的）
            print(
                f"step {step:6d} | train loss {loss.item():.4f} | "
                f"val loss {val_loss.item():.4f} | lr {cur_lr:.2e} | 累计用时 {dt:6.1f}s"
            )
            # loss.item() 把单元素张量转成 Python 数字
            # :6d 表示占 6 位右对齐，:.4f 表示保留 4 位小数

            # 把这一步的 loss 写进日志（plot_loss.py 画图用）
            if log_file is not None:
                log_file.write(
                    f"{step},{loss.item():.6f},{val_loss.item():.6f},{dt:.1f}\n"
                )
                log_file.flush()
                # ↑ flush() 立即写入磁盘，防止程序中断丢数据

    # ==========================================================================
    # 【第 6 步：保存模型】
    # ==========================================================================
    torch.save(
        {
            "model": model.state_dict(),   # 模型的所有权重（一个字典）
            "stoi": stoi,                  # 字符→id 映射（生成时需要）
            "itos": itos,                  # id→字符 映射（生成时需要）
            "vocab_size": vocab_size,      # 词表大小
            "config": vars(args),          # 训练时用的所有超参数（重建模型需要）
        },
        args.save,
    )
    # ↑ torch.save 把一个 Python 对象序列化成二进制文件存到磁盘。
    #   state_dict() 返回模型所有权重的字典（key 是层名，value 是权重张量）。
    #   我们同时存了词表和配置，这样 generate.py 加载时能完整重建模型。
    print(f"训练完成，模型已保存到 {args.save}")

    if log_file is not None:
        log_file.close()  # 关闭日志文件

    # ==========================================================================
    # 【第 7 步：打印每一层的权重（让你看到"门道"）】
    # ==========================================================================
    if args.print_weights:
        print("\n===== 每一层的权重（这就是训练出来的数字）=====")
        # named_parameters() 遍历模型里所有可训练参数，返回 (名字, 张量) 对
        for name, p in model.named_parameters():
            print(
                f"{name:44s} shape={str(list(p.shape)):18s} "
                f"参数量={p.numel():>8,}  均值={p.mean().item():+.4f}  标准差={p.std().item():.4f}"
            )
            # name:44s  名字占 44 位左对齐
            # p.shape   权重张量的形状（比如 [384, 128] 表示 384 行 128 列）
            # p.numel() 权重里数字的总个数（numel = number of elements）
            # p.mean()  所有权重的平均值（训练后应该接近 0，因为权重以 0 为中心分布）
            # p.std()   所有权重的标准差（衡量权重分散程度，太大会不稳定）


if __name__ == "__main__":
    # ↑ 这是 Python 的标准写法：只有当这个文件被"直接运行"时才执行 main()，
    #   如果被其他文件 import 则不执行（防止 import 时自动跑训练）。
    main()
