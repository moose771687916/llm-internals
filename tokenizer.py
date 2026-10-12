# tokenizer.py
# 完整 BPE 分词器：训练（学合并规则）+ 编码（文本→id）+ 解码（id→文本）
# 这就是 GPT-2 / GPT-4 / Llama / DeepSeek 里 tokenizer 的简化教学版
#
# 三个方法对应三个阶段：
#   train()  —— 在数据上学"合并规则"（哪两个相邻片段最常挨着 → 合并成一个新 token）
#   encode() —— 把一段文本切成 token id 序列（模型吃的东西）
#   decode() —— 把 token id 序列还原成文本（模型吐的东西）
from collections import Counter
import pickle


class BPETokenizer:
    def __init__(self):
        self.merges = {}       # 合并规则：{(旧id, 旧id): 新id}
        self.vocab = {}        # 词表：{id: 文本片段}（解码用）
        self.char_to_id = {}   # 初始字符 → id（编码用）
        self.num_initial = 0   # 初始字符有多少个（新 token 从这之后编号）

    # ---------- 第 1 步：训练 ----------
    def train(self, text, num_merges):
        """在数据上学合并规则，合并 num_merges 次"""
        # 初始：每个字符一个 token
        chars = sorted(set(text))
        self.num_initial = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        next_id = len(chars)  # 新 token 从 num_initial 开始编号

        # 先把整个数据转成 id 序列
        ids = [self.char_to_id[c] for c in text]
        tokens = ids  # 训练过程中这个序列会越来越短（不断被合并）

        for k in range(num_merges):
            # 数一数：当前序列里哪个"相邻对"出现次数最多
            stats = Counter(zip(tokens, tokens[1:]))
            if not stats:
                break
            pair, count = max(stats.items(), key=lambda x: x[1])

            # 合并它：记规则 + 实际执行合并
            self.merges[pair] = next_id
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            tokens = self._merge_ids(tokens, pair, next_id)
            next_id += 1

            # 打印进度（前 10 次 + 之后每 100 次）
            if k < 10 or (k + 1) % 100 == 0:
                print(f"  合并 {k+1:4d} 次：{self.vocab[pair[0]]!r}+{self.vocab[pair[1]]!r}"
                      f" → #{next_id-1}（出现 {count} 次）")

        print(f"训练完成：词表 = {next_id}（{self.num_initial} 个初始字符 + {num_merges} 次合并）")
        return self

    # ---------- 第 2 步：编码 ----------
    def encode(self, text):
        """文本 → token id 列表（按合并规则的学习顺序应用）"""
        ids = [self.char_to_id[c] for c in text]
        # 合并规则按"学习顺序"（新 id 从小到大）应用：
        # 先学的规则（编号小）优先，因为它是更基础/更高频的合并
        for pair, new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            ids = self._merge_ids(ids, pair, new_id)
        return ids

    # ---------- 第 3 步：解码 ----------
    def decode(self, ids):
        """token id 列表 → 文本"""
        return "".join(self.vocab[i] for i in ids)

    # ---------- 内部工具 ----------
    def _merge_ids(self, ids, pair, new_id):
        """把序列里所有 (pair[0], pair[1]) 相邻对替换成 new_id"""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    # ---------- 保存 / 加载 ----------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab,
                         "char_to_id": self.char_to_id, "num_initial": self.num_initial}, f)

    @staticmethod
    def load(path):
        t = BPETokenizer()
        with open(path, "rb") as f:
            d = pickle.load(f)
        t.merges, t.vocab, t.char_to_id, t.num_initial = (
            d["merges"], d["vocab"], d["char_to_id"], d["num_initial"])
        return t


# ---------- 主入口：在莎士比亚数据上训练并保存 ----------
if __name__ == "__main__":
    import urllib.request

    print("下载莎士比亚数据...")
    data = urllib.request.urlopen(
        "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
    ).read().decode("utf-8")
    print(f"数据：{len(data):,} 字符\n")

    print("开始 BPE 训练（词表目标：2,000）：")
    tok = BPETokenizer()
    tok.train(data, num_merges=2000)
    tok.save("bpe_tokenizer.pkl")

    # 验证：编码 → 解码 能完整还原吗？
    sample = data[:2000]
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"\n还原测试：编码 {len(sample)} 字符 → {len(ids)} token → 解码还原一致？{back == sample}")

    # 看一眼词表里"长出来的词"
    print("\n词表抽样（合并出来的最长片段）：")
    merges_sorted = sorted(tok.merges.items(), key=lambda kv: kv[1])
    for (a, b), new_id in merges_sorted[-15:]:
        print(f"  #{new_id:4d} = {tok.vocab[a]!r} + {tok.vocab[b]!r} → {tok.vocab[new_id]!r}")
