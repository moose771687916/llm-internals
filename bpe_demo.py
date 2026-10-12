# bpe_demo.py
# 用真实莎士比亚数据演示 BPE（字节对编码，Byte Pair Encoding）的合并过程
# GPT-2 / GPT-4 / Llama / DeepSeek 的 tokenizer 核心思想就是这个
from collections import Counter
import urllib.request

print("=" * 60)
print("BPE 演示：从'每个字符一个 token'开始，反复合并最高频的相邻对")
print("=" * 60)

# 1. 下载数据（前 20000 字符就够演示）
data = urllib.request.urlopen(
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
).read().decode("utf-8")[:20000]

# 2. 初始：每个字符是一个 token
tokens = list(data)
print(f"\n数据：{len(data):,} 个字符，共 {len(set(tokens))} 种不同字符")
print(f"初始 token 数：{len(tokens):,}（1 字符 = 1 token）\n")


# 3. 统计相邻字符对
def get_stats(tokens):
    return Counter(zip(tokens, tokens[1:]))


# 4. 把某个相邻对合并成一个新 token
def merge(tokens, pair, new_token):
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_token)  # 合并成一个新 token
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


# 5. 循环合并 N 次：每次都找"出现次数最多的相邻对"合并
N = 50
merges = {}  # 记录所有合并规则：{(a, b): 新token编号}
for k in range(N):
    stats = get_stats(tokens)
    if not stats:
        break
    top_pair, count = max(stats.items(), key=lambda x: x[1])
    new_token = 256 + k  # 新 token 的编号（256 之前留给原始字符）
    tokens = merge(tokens, top_pair, new_token)
    merges[top_pair] = new_token
    if k < 12:
        print(f"  第 {k+1:2d} 次合并：{top_pair[0]!r} + {top_pair[1]!r} → token #{new_token}（这对出现 {count} 次）")

print(f"\n合并 {N} 次后：{len(tokens):,} 个 token（压缩比 {len(data)/len(tokens):.2f} 倍）")

# 6. 构建词表：字符 token → 文本，合并 token → 拼接文本
vocab = {}
for ch in set(data):
    vocab[ch] = ch
for (a, b), new_id in merges.items():
    vocab[new_id] = vocab[a] + vocab[b]  # 新 token 的内容 = 两个旧 token 拼起来

print(f"\n词表大小：{len(vocab)}（{len(set(data))} 个原始字符 + {len(merges)} 个合并 token）")
print("\n学到的合并规则前 20 条（看看模型'发明'了什么）：")
for i, ((a, b), new_id) in enumerate(list(merges.items())[:20]):
    print(f"  #{new_id:3d} = {vocab[a]!r} + {vocab[b]!r} → {vocab[new_id]!r}")

# 7. 展示合并后的 token 流（前 150 个），直观看到高频片段被合成了一个 token
print("\n合并后的 token 流（前 150 个 token，编号 < 256 是原始字符）：")
print("  " + " ".join(str(t) if isinstance(t, str) else f"[{t}]" for t in tokens[:150]))
