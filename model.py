# -*- coding: utf-8 -*-
"""
================================================================================
model.py —— 迷你 GPT 模型的"设计图纸"
================================================================================

【这个文件是干嘛的】
    它定义了我们的神经网络"长什么样"。就像造房子先画图纸一样，
    这个文件只描述"模型有几层、每层有多少个神经元、数据怎么流动"，
    它本身不训练、不生成，是被 train.py（训练）和 generate.py（生成）
    调用的"零件库"。

【阅读顺序建议（新手从这里开始）】
    1. 先看最下面的 GPT 类 —— 它是整个模型的"总装图"，一眼能看懂全貌
    2. 再看 Block 类 —— GPT 就是把 Block 重复堆了好几层
    3. 然后看 CausalSelfAttention —— 这是 GPT 最核心的"注意力机制"
    4. 最后看 LayerNorm / MLP —— 辅助零件

【本文件用到的核心概念速查】
    张量(tensor)：可以理解成"多维数组"。0维是一个数，1维是一行数，
                  2维是一张表（矩阵），3维是一摞表。PyTorch 里所有数据都是张量。
    形状(shape)：张量每个维度有多长。比如 shape=[64, 128] 表示 64 行 128 列。
    参数(parameter)/权重(weight)：模型里"需要被训练的数字"。训练前是随机数，
                  训练后变成"有意义的数字"。整个模型有 82 万个这样的数字。
    前向传播(forward)：数据从输入层一路算到输出层的过程。
    B/T/C：代码里经常出现的缩写。B=batch（一批几条数据），T=time/序列长度
                  （一条数据有几个字符），C=channel/特征维度（每个字符用多少个数表示）。
"""

# ==============================================================================
# 【import 部分：导入别人写好的工具】
# ==============================================================================
# import 就是"把别人写好的代码拿过来用"，不用自己从零造轮子。

import torch
# ↑ PyTorch 是 Facebook 开源的深度学习库，是整个项目的地基。
#   它提供了：张量运算（加减乘除、矩阵乘法）、自动求导（backward）、
#   随机数、GPU 加速等所有底层能力。没有它，我们要自己写几万行 C++。

import torch.nn as nn
# ↑ torch.nn 是 PyTorch 里专门做"神经网络"的子模块，我们给它起个别名叫 nn（短一点）。
#   里面有现成的"积木"：nn.Linear（全连接层）、nn.Embedding（嵌入层）、
#   nn.Dropout（随机丢弃层）等。我们的模型就是用这些积木搭出来的。

from torch.nn import functional as F
# ↑ functional 是 nn 的"函数版"兄弟模块，别名 F。
#   区别：nn 里的东西是"类"（有状态，会记住权重，比如 Linear 层有权重）；
#         F 里的东西是"纯函数"（无状态，调用一次算一次，比如 softmax、cross_entropy）。
#   我们用 F 来做 softmax（归一化）、cross_entropy（算损失）这些"一次性运算"。

torch.manual_seed(1337)
# ↑ 设置随机数种子。神经网络的权重初始值是随机的，设置种子后，
#   每次运行生成的随机数都一样，保证结果可复现（你和我跑出来的结果一致）。
#   1337 是随便选的一个数字，换成任何整数都行。


# ==============================================================================
# 【零件 1：LayerNorm —— 层归一化】
# ==============================================================================
# 作用：让每一层输出的数值"不要太离谱"，保持在一个稳定的范围内。
# 为什么需要：神经网络很深的时候，数据一层层传下去，数值可能越来越大或越来越小，
#             导致训练不稳定（loss 震荡甚至不下降）。LayerNorm 像一个"稳压器"。
# 通俗理解：每次考完试把全班分数标准化一下（减均值除标准差），让分数分布稳定。

class LayerNorm(nn.Module):
    # ↑ nn.Module 是 PyTorch 里所有神经网络模块的"祖宗类"。
    #   继承它之后，我们的类就自动拥有了：记录参数、保存模型、.to(device) 等能力。

    def __init__(self, dim, eps=1e-5):
        # ↑ __init__ 是类的"构造函数"，创建这个对象时自动调用。
        #   dim：输入数据的特征维度（比如 128，就是每个位置有 128 个数）
        #   eps：一个很小的数（0.00001），防止除以 0
        super().__init__()
        # ↑ 调用父类 nn.Module 的初始化，必须写，否则一些内部机制不工作。
        self.eps = eps
        # gamma 和 beta 是 LayerNorm 里"可学习的参数"（也是权重的一种）
        # 归一化后，模型可以通过这两个参数"选择要不要恢复原来的尺度"
        self.gamma = nn.Parameter(torch.ones(dim))
        # ↑ ones(dim) 生成一个全是 1 的一维数组，长度为 dim。
        #   nn.Parameter(...) 把它包装成"可训练参数"，PyTorch 会自动在训练时更新它。
        self.beta = nn.Parameter(torch.zeros(dim))
        # ↑ 全是 0 的数组，作为可训练的"偏移量"。

    def forward(self, x):
        # ↑ forward 定义了"数据进来后怎么算出去"。这是 nn.Module 的核心方法。
        #   调用 model(x) 时，PyTorch 会自动调用这个 forward 方法。
        #   x 的形状：(B, T, dim)，一批数据，每条 T 个位置，每个位置 dim 个数
        mean = x.mean(-1, keepdim=True)
        # ↑ 对最后一个维度（dim 那个维度）求均值。
        #   -1 表示最后一维；keepdim=True 保持维度不变，方便后面广播计算。
        #   结果形状：(B, T, 1)
        var = x.var(-1, keepdim=True, unbiased=False)
        # ↑ 求方差（每个数和均值差的平方的平均）。unbiased=False 表示用总体方差。
        x = (x - mean) / torch.sqrt(var + self.eps)
        # ↑ 归一化：减均值、除标准差。加 eps 防止方差为 0 时除以 0。
        return self.gamma * x + self.beta
        # ↑ 用可学习的 gamma 缩放、beta 偏移。模型可以学到"要不要归一化、归一化多少"。


# ==============================================================================
# 【零件 1.5：RoPE 旋转位置编码（现代大模型同款）】
# ==============================================================================
# 绝对位置表（零件 2 的 GPT 类里）是"查表"：把位置当字符查向量。
# RoPE 不是查表，而是把 Q/K 向量本身"按位置转个角度"——位置信息写进向量的方向里。
# 关键性质：点积只看"相对距离"（|i-j|），不看"第几位"，所以可以外推到
# 训练时没见过的新位置。旋转公式没有任何可学习参数（0 参数，纯数学规则）。

def apply_rope(q, k, cos, sin):
    """把 Q、K 的最后一维（head_dim）按位置旋转 —— RoPE 核心。
    q/k: (B, n_head, T, head_dim)；cos/sin: (1, 1, T, head_dim//2) 每对"钟"的角度
    """
    shape = q.shape
    # 把 head_dim 拆成 head_dim//2 对（每对 2 个数 = 一个"钟"）
    # 第 0 对 = (数字0, 数字1)，第 1 对 = (数字2, 数字3)，……
    q = q.view(*shape[:-1], -1, 2)   # (B, n_head, T, d/2, 2)
    k = k.view(*shape[:-1], -1, 2)
    q0, q1 = q[..., 0], q[..., 1]    # 每对的第 1 个数、第 2 个数
    k0, k1 = k[..., 0], k[..., 1]
    # 2 维旋转公式（和初中三角函数一模一样）：
    #   x' = x·cosθ − y·sinθ
    #   y' = x·sinθ + y·cosθ
    # 这里 θ 是"位置 × 转速"（cos/sin 已算好），广播到每个 batch 和每个头
    q0n = q0 * cos - q1 * sin
    q1n = q0 * sin + q1 * cos
    k0n = k0 * cos - k1 * sin
    k1n = k0 * sin + k1 * cos
    # 把旋转后的两半拼回原来的形状
    q = torch.stack((q0n, q1n), dim=-1).view(shape)
    k = torch.stack((k0n, k1n), dim=-1).view(shape)
    return q, k


# ==============================================================================
# 【零件 2：CausalSelfAttention —— 带掩码的多头自注意力（GPT 的心脏）】
# ==============================================================================
# 这是整个 GPT 最核心、最难理解的部分。我用大白话讲清楚：
#
# 【自注意力在干什么】
#   想象你在读一句话："我吃了一个苹果"。当你读到"苹果"时，你会自动回想
#   前面的"吃"和"一个"，因为它们和"苹果"有关系。自注意力就是让模型做这件事：
#   每个字符都去"看"前面所有字符，判断谁和自己最相关，然后把相关字符的信息
#   加权汇总到自己身上。
#
# 【Q / K / V 是什么】
#   Query（查询）："我现在这个字符想找什么信息"——比如"苹果"想找"被吃的东西"
#   Key（键）：每个字符挂出的"名片"——"我是吃""我是一个""我是苹果"
#   Value（值）：每个字符实际携带的"内容信息"
#   计算过程：用我的 Q 去和所有人的 K 比相似度 → 相似度变成权重 → 用权重对所有人的 V 加权求和
#
# 【多头（multi-head）是什么】
#   不是只做一次注意力，而是分成好几组（比如 4 组），每组独立学一种"关注模式"。
#   有的头可能学"关注主语"，有的头学"关注动词"，有的头学"关注标点"。
#   最后把各组的结果拼起来。
#
# 【因果掩码（causal mask）是什么】
#   预测第 5 个字符时，只能看前 4 个字符，不能偷看第 5 个及以后（那是答案）。
#   掩码就是一个上三角矩阵，把"未来位置"的注意力分数设成负无穷，softmax 后变成 0。

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, pos_enc="abs", rope_base=10000.0):
        # ↑ n_embd：每个字符用多少个数表示（特征维度，比如 128）
        #   n_head：分成几个头（比如 4）
        #   block_size：上下文最大长度（一次最多看多少个字符，比如 128）
        #   dropout：随机丢弃比例（训练时随机把一些神经元置 0，防止过拟合）
        #   pos_enc：位置编码方式 "abs"=绝对位置表（GPT-2 经典）/ "rope"=旋转位置编码（现代大模型）
        #   rope_base：RoPE 转速公式里的 base（默认 10000；Qwen 训练时把它调到百万级）
        super().__init__()
        assert n_embd % n_head == 0
        # ↑ 断言：特征维度必须能被头数整除，因为要平均分给每个头。
        #   比如 128 维 / 4 头 = 每个头 32 维。
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 每个头分到的维度
        self.pos_enc = pos_enc

        # ---- 核心权重矩阵 ----
        # 一个线性层同时算出 Q、K、V（把三个矩阵拼在一起，效率更高）
        # 输入 n_embd 维，输出 3*n_embd 维（前 1/3 是 Q，中间 1/3 是 K，后 1/3 是 V）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # ↑ nn.Linear(in, out) 是一个"全连接层"，内部有一个权重矩阵 shape=[out, in]。
        #   它做的事就是：输出 = 输入 @ 权重矩阵.T（矩阵乘法）。
        #   bias=False 表示不要偏置项（只做矩阵乘法，不加常数）。

        # 把多头拼回来之后，再做一次线性变换，融合各头的信息
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)
        # ↑ 随机丢弃层：训练时随机把一部分输入置 0，强制模型不要依赖任何单个神经元，
        #   防止"死记硬背"（过拟合）。推理时自动不生效。

        # ---- 因果掩码 ----
        # register_buffer 注册一个"不是参数但要跟着模型走"的张量（不会被训练，但保存模型时会一起存）
        # tril 是下三角矩阵：对角线及以下是 1，以上是 0
        # 形状：(1, 1, block_size, block_size)，前面两个 1 是为了广播到 B 和 n_head 维度
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        # ---- RoPE 频率表（pos_enc="rope" 时才需要）----
        # 预先把每个位置的 cos/sin 角度算好存起来（buffer，不参与训练），
        # 前向时直接查表用，省得每次现算三角函数。
        if pos_enc == "rope":
            # 转速公式：θ_i = rope_base^(-2i/head_dim)，i = 0..head_dim/2-1
            # 第 0 个"钟"转最快（θ=1.0 弧度/步），最后 1 个"钟"转最慢
            # base 越大 → 所有钟转速越慢 → 角度分布越平缓（Qwen 训练时把 base 调到 100 万）
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            # ↑ head_dim=32 时：arange(0,32,2)=[0,2,4,...,30]
            #   10000^(-2i/32) = [1.0, 0.75, 0.56, ...] —— 从快到慢的 16 个转速
            t = torch.arange(block_size).float().unsqueeze(1) * inv_freq.unsqueeze(0)
            # ↑ 位置 × 转速 → 每个位置、每个"钟"转了多少弧度。(block_size, head_dim/2)
            #   视角 1 转 1θ，位置 2 转 2θ……和讲过的"秒针/年钟"完全对应
            self.register_buffer("cos", t.cos().view(1, 1, block_size, -1))  # (1,1,T,d/2)
            self.register_buffer("sin", t.sin().view(1, 1, block_size, -1))

    def forward(self, x):
        # ↑ x 的形状：(B, T, C)，C = n_embd
        B, T, C = x.shape  # 解包出三个维度的大小

        # ---- 第一步：算出 Q、K、V ----
        qkv = self.c_attn(x)          # (B, T, 3C)，一次算出 QKV 拼在一起
        q, k, v = qkv.split(C, dim=2) # 沿最后一维切成三份，各 (B, T, C)

        # ---- 第二步：拆成多头 ----
        # 把 (B, T, C) 变成 (B, n_head, T, head_dim)
        # view 是"改变形状"（不改变数据），transpose 是"交换两个维度"
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # 现在 q/k/v 的形状都是 (B, n_head, T, head_dim)

        # ---- 第二步半：RoPE 旋转（pos_enc="rope" 时）----
        # 在算点积之前，把 Q 和 K 按各自位置旋转（V 不转——它只是内容，不参与"找谁相关"）。
        # 旋转之后点积就带着"相对距离"信号（cos 偶函数：只认距离不认方向，方向由掩码管）。
        if self.pos_enc == "rope":
            q, k = apply_rope(q, k, self.cos[:, :, :T, :], self.sin[:, :, :T, :])

        # ---- 第三步：算注意力分数 ----
        # 注意力分数 = Q · K^T / sqrt(head_dim)
        # 为什么除以 sqrt(head_dim)：防止维度大时点积结果太大，导致 softmax 后梯度消失
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # ↑ @ 是矩阵乘法。q 是 (B, n_head, T, head_dim)，k.transpose 是 (B, n_head, head_dim, T)
        #   结果形状：(B, n_head, T, T) —— 每个位置对每个位置的注意力分数

        # ---- 第四步：应用因果掩码 ----
        # 把"未来位置"（上三角）的分数设成负无穷，这样 softmax 后权重就是 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # ---- 第五步：softmax 归一化成权重 ----
        att = F.softmax(att, dim=-1)  # 对最后一维做 softmax，每行加起来等于 1
        att = self.dropout(att)        # 训练时随机丢弃一些注意力连接

        # ---- 第六步：用权重对 V 加权求和 ----
        y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_dim) = (B, n_head, T, head_dim)

        # ---- 第七步：把多头拼回来 ----
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # ↑ transpose 交换维度后，内存可能不连续，contiguous() 让它连续，才能 view
        #   最终形状回到 (B, T, C)

        return self.c_proj(y)  # 最后一次线性变换，融合各头信息


# ==============================================================================
# 【零件 3：MLP —— 前馈网络（多层感知机）】
# ==============================================================================
# 作用：对每个位置的特征做一次"非线性变换"，增加模型的表达能力。
# 为什么需要：注意力层负责"在不同字符之间交换信息"，MLP 负责"在每个字符内部
#             深度加工信息"。两者交替出现，是 Transformer 的标准结构。
# 结构：线性层(放大4倍) → GELU激活 → 线性层(缩回来) → Dropout
# 放大 4 倍是 GPT 论文里的标准配置，让中间层有更多容量去学复杂模式。

class MLP(nn.Module):
    # ↑ MLP = Multi-Layer Perceptron（多层感知机），也叫"前馈网络"。
    #   作用：每个字符"独立思考"——拿到注意力交流的结果后，自己深加工。
    #   activation 参数支持两种：
    #     "gelu"  （默认）：一条路，Linear→GELU→Linear（经典 GPT 版）
    #     "swiglu"       ：两条路，门控（SiLU×值）→Linear（Llama/DeepSeek 等现代大模型版）
    #   SwiGLU 的中间维度用 8/3×n_embd，是为了让参数量和 GELU 版基本持平（消融实验要公平对比）。

    def __init__(self, n_embd, dropout=0.0, activation="gelu"):
        super().__init__()
        self.activation = activation

        if activation == "gelu":
            self.net = nn.Sequential(
                # ↑ Sequential 把多个层按顺序串起来，数据依次流过
                nn.Linear(n_embd, 4 * n_embd),
                # ↑ 第一层：把维度从 n_embd 放大到 4 倍。有偏置（默认 bias=True）
                nn.GELU(),
                # ↑ GELU 是激活函数，给模型引入"非线性"。
                #   没有激活函数的话，多层线性层叠起来还是线性的，学不到复杂规律。
                #   GELU ≈ 输入为正时近似等于输入，为负时缓慢衰减（比 ReLU 平滑）。
                nn.Linear(4 * n_embd, n_embd),
                # ↑ 第二层：把维度从 4 倍缩回来
                nn.Dropout(dropout),
                # ↑ 随机丢弃，防过拟合
            )
        elif activation == "swiglu":
            # SwiGLU = Swish + Gated Linear Unit（Swish 门控线性单元）
            # 两条路：门路（过 SiLU 激活）× 值路（不过激活），逐元素相乘，再压回。
            # 类比：一条路决定"放多少信息"（门），另一条路提供"信息内容"（值）。
            hidden = int(8 * n_embd / 3)  # ≈341，让参数量与 GELU 版持平
            self.w_gate = nn.Linear(n_embd, hidden, bias=False)  # 门路（要过激活）
            self.w_value = nn.Linear(n_embd, hidden, bias=False)  # 值路（不过激活）
            self.w_out = nn.Linear(hidden, n_embd, bias=False)   # 压回原维度
            self.act = nn.SiLU()  # SiLU ≈ GELU 的兄弟（平滑激活）
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError(f"未知激活函数: {activation}（支持 gelu / swiglu）")

    def forward(self, x):
        if self.activation == "gelu":
            return self.net(x)  # 数据依次流过上面四个层
        else:  # swiglu
            # 门路输出 × 值路输出 → 压回 → dropout
            return self.drop(self.w_out(self.act(self.w_gate(x)) * self.w_value(x)))


# ==============================================================================
# 【零件 4：Block —— 一个 Transformer 层】
# ==============================================================================
# 一个 Block = 注意力 + MLP，各套一层 LayerNorm，外面接残差连接。
# GPT 就是把这个 Block 重复堆 n_layer 次（比如 4 次、8 次、96 次）。
#
# 【残差连接（residual connection）是什么】
#   x = x + 子层(x) —— 把子层的输出"加回"原始输入上。
#   好处：信息可以直接"抄近路"穿过深层网络，梯度反向传播时也不容易消失，
#         让几十层甚至上百层的网络也能训练。这是深度学习的关键发明之一。

class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, activation="gelu", pos_enc="abs", rope_base=10000.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)       # 注意力前的归一化
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, pos_enc, rope_base)  # 注意力
        self.ln2 = LayerNorm(n_embd)       # MLP 前的归一化
        self.mlp = MLP(n_embd, dropout, activation)  # 前馈网络（gelu 或 swiglu）

    def forward(self, x):
        # 先归一化 → 注意力 → 残差加回来
        x = x + self.attn(self.ln1(x))
        # 再归一化 → MLP → 残差加回来
        x = x + self.mlp(self.ln2(x))
        return x


# ==============================================================================
# 【总装：GPT —— 完整的迷你 GPT 模型】
# ==============================================================================
# 数据流向：
#   输入字符id → 词嵌入(变成向量) + 位置嵌入(加上位置信息)
#   → 经过 n_layer 个 Block(注意力+MLP 重复)
#   → 最后归一化 → 输出层(映射回词表概率)
#   → 每个位置输出"下一个字符是谁"的概率分布

class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.0, activation="gelu", pos_enc="abs", rope_base=10000.0):
        # ↑ vocab_size：词表大小（有多少种不同的字符，我们的莎士比亚数据是 65）
        #   n_embd：每个字符用多少个数表示（向量维度）
        #   n_head：注意力头数
        #   n_layer：堆多少个 Transformer 层（这就是"权重的层数"）
        #   block_size：上下文最大长度
        #   activation：MLP 激活函数（"gelu" 经典版 / "swiglu" 现代大模型版）
        #   pos_enc：位置编码（"abs"=绝对位置表 GPT-2 经典 / "rope"=旋转位置编码 现代大模型）
        #   rope_base：RoPE 转速公式的 base（默认 10000；Qwen 训练时调大到百万级支持长上下文）
        super().__init__()
        self.block_size = block_size
        self.pos_enc = pos_enc

        # ---- 嵌入层 ----
        # 词嵌入：把每个字符 id 映射成一个 n_embd 维的向量
        # 内部是一个权重矩阵 shape=[vocab_size, n_embd]，每行是一个字符的向量
        # 训练过程中，这些向量会逐渐学到"字符的含义"（比如 'a' 和 'b' 的向量会不同）
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        # 位置编码：两种方式二选一
        # "abs"：位置嵌入表。给每个"位置"一个向量（GPT-2 经典做法）。
        #        内部权重矩阵 shape=[block_size, n_embd]，每行是一个位置的向量
        # "rope"：不建表！RoPE 的旋转公式没有任何可学习参数，
        #         cos/sin 表在注意力层里用公式算（0 参数，还能外推到新位置）
        if pos_enc == "abs":
            self.position_embedding = nn.Embedding(block_size, n_embd)

        # ---- 堆叠 n_layer 个 Transformer 层 ----
        # ModuleList 是 PyTorch 专门放"一堆子模块"的容器，会自动注册里面的参数
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, activation, pos_enc, rope_base)
             for _ in range(n_layer)]
        )

        # ---- 最后的归一化和输出层 ----
        self.ln_f = LayerNorm(n_embd)  # 最后一层归一化
        # 输出层：把 n_embd 维的隐藏向量映射回 vocab_size 维（每个字符一个分数）
        # 这个分数经过 softmax 就变成"下一个字符是各个字符的概率"
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # ---- 初始化所有权重 ----
        self.apply(self._init_weights)
        # ↑ apply 会递归遍历模型里所有子模块，对每个调用 _init_weights

    def _init_weights(self, module):
        """初始化权重：训练前给所有参数一个合适的随机起点"""
        if isinstance(module, nn.Linear):
            # 线性层的权重：正态分布，均值 0，标准差 0.02
            # 0.02 是 GPT 论文里的标准值，太小梯度消失，太大训练不稳定
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # 偏置初始化为 0
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx：输入的字符 id 序列，形状 (B, T)
        targets：可选，训练时传入的"正确答案"，形状 (B, T)，每个位置是下一个字符的 id
        返回：(logits, loss)
              logits：每个位置预测下一个字符的"分数"，形状 (B, T, vocab_size)
              loss：预测和正确答案的差距，训练时才有
        """
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过最大上下文 {self.block_size}"

        # ---- 第一步：嵌入 ----
        tok = self.token_embedding(idx)              # (B, T, n_embd)，每个字符变成向量
        if self.pos_enc == "abs":
            # 绝对位置表：字符向量 + 位置向量（位置信息靠"加"进去）
            pos = self.position_embedding(
                torch.arange(T, device=idx.device)
            )                                        # (T, n_embd)，每个位置的向量
            x = tok + pos                            # 字符向量 + 位置向量 = 带位置信息的输入
        else:
            # RoPE：不加法！位置信息由注意力层里对 Q/K 的"旋转"提供（见 CausalSelfAttention）
            x = tok

        # ---- 第二步：流过所有 Transformer 层 ----
        for blk in self.blocks:
            x = blk(x)

        # ---- 第三步：最终归一化 + 输出层 ----
        x = self.ln_f(x)
        logits = self.head(x)                         # (B, T, vocab_size)

        # ---- 第四步：算损失（训练时） ----
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # cross_entropy 把分数变成概率，再和正确答案比，算出"差距"（一个数）
            # 差距越小，模型预测越准。训练的目标就是让这个数越来越小。
            loss = F.cross_entropy(
                logits.view(B * T, C),   # 把 (B,T,C) 展平成 (B*T, C)
                targets.view(B * T),      # 把 (B,T) 展平成 (B*T,)
            )
        return logits, loss

    @torch.no_grad()
    # ↑ 这个装饰器告诉 PyTorch：下面的代码不需要计算梯度（推理时省内存、加速）
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        给定开头 idx，逐字预测下一个字符，循环生成 max_new_tokens 个字符。
        idx：开头序列，形状 (1, T)（batch 为 1，一次只生成一条）
        max_new_tokens：要生成多少个新字符
        temperature：温度，控制随机性。越小越保守（总选概率最大的），越大越随机
        """
        for _ in range(max_new_tokens):
            # 只取最近的 block_size 个字符（模型一次最多看这么多）
            idx_cond = idx[:, -self.block_size:]

            # 前向传播，得到每个位置的分数
            logits, _ = self(idx_cond)

            # 只取最后一个位置的分数（我们要预测的是"下一个字符"）
            logits = logits[:, -1, :] / temperature
            # 除以 temperature：temperature<1 时分数差距被放大（更确定），>1 时被抹平（更随机）

            # softmax 把分数变成概率（加起来等于 1）
            probs = F.softmax(logits, dim=-1)

            # 按概率抽样：不是每次都选概率最大的，而是按概率随机选（这样才有多样性）
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 把新字符拼到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def count_parameters(self):
        """统计模型总参数量（权重数字的总个数）"""
        return sum(p.numel() for p in self.parameters())
