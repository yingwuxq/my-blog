# 注意力机制全系列

## 概述

注意力机制（Attention Mechanism）是深度学习中最重要的创新之一。其核心思想是**允许模型在处理某个位置时动态地关注输入序列中其他位置的相关信息**，而不是将所有信息均匀压缩到一个固定长度的向量中。

本章系统梳理从基础注意力到现代高效注意力变体的完整谱系。

---

## 1. Scaled Dot-Product Attention（缩放点积注意力）

这是所有现代注意力机制的**原子操作**。

### 定义

给定三个矩阵：查询（Query）$Q$、键（Key）$K$、值（Value）$V$，注意力输出定义为值的加权和：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$。

### 物理意义

- **Query**（查询）：当前"问问题"的位置
- **Key**（键）：所有位置的"标签"，与 Query 计算相似度
- **Value**（值）：所有位置的"内容"
- **Attention Weight**（注意力权重）：Query 对每个 Key 的关注程度（概率分布）
- **Output**（输出）：所有 Value 的加权和

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    标准 Scaled Dot-Product Attention
    
    Args:
        Q: [batch, heads, seq_q, d_k]
        K: [batch, heads, seq_k, d_k]  
        V: [batch, heads, seq_v, d_v]  (seq_v = seq_k)
        mask: [batch, 1, seq_q, seq_k] (可选，-inf 掩码)
    Returns:
        output: [batch, heads, seq_q, d_v]
        attention_weights: [batch, heads, seq_q, seq_k]
    """
    d_k = Q.size(-1)
    
    # Q @ K^T: [batch, heads, seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 梯度分析

Softmax Attention 的一个有趣特性是梯度形式简洁。设 $s_i = \frac{q \cdot k_i}{\sqrt{d_k}}$，$a_i = \text{softmax}(s)_i$，输出为 $\sum a_i v_i$，则对 $s_j$ 的梯度为：

$$\frac{\partial \text{output}}{\partial s_j} = a_j (v_j - \text{output})$$

这意味着：
- 如果 $v_j$ 高于当前加权平均 $\text{output}$，梯度为正（提高该位置的注意力权重）
- 如果 $v_j$ 低于当前加权平均，梯度为负（降低该位置的注意力权重）

---

## 2. 自注意力（Self-Attention）与交叉注意力（Cross-Attention）

### 自注意力（Self-Attention）

当 Q、K、V 全部来自同一个序列时，称为自注意力（Self-Attention，也称 Intra-Attention）：

$$\text{Self-Attention}(X) = \text{Attention}(XW^Q, XW^K, XW^V)$$

**特点**：序列中每个位置与所有其他位置计算关联。Encoder 层的 MHA 即是自注意力。

### 交叉注意力（Cross-Attention）

当 Q 来自一个序列，而 K、V 来自另一个序列时，称为交叉注意力（Cross-Attention，也称 Inter-Attention）：

$$\text{Cross-Attention}(X_{\text{dec}}, X_{\text{enc}}) = \text{Attention}(X_{\text{dec}}W^Q, X_{\text{enc}}W^K, X_{\text{enc}}W^V)$$

**特点**：Decoder 层的注意力即是交叉注意力——Q 来自 Decoder，K、V 来自 Encoder，使得 Decoder 在生成时参考输入序列的信息。

```python
# 自注意力示例
x = torch.randn(2, 10, 512)  # [batch, seq_len, d_model]
self_attn_out = scaled_dot_product_attention(x, x, x)

# 交叉注意力示例
encoder_out = torch.randn(2, 15, 512)  # Encoder 输出
decoder_input = torch.randn(2, 10, 512)  # Decoder 输入
cross_attn_out = scaled_dot_product_attention(decoder_input, encoder_out, encoder_out)
```

---

## 3. 多头注意力（Multi-Head Attention, MHA）

### 原理

多头注意力将模型拆分为 $h$ 个独立的注意力头，每个头在不同的特征子空间中学习注意力模式：

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**直觉**：不同的头可以关注不同的关系类型——比如在翻译任务中，一个头关注语法依赖，另一个头关注语义相似性。

### 参数量

MHA 的投影矩阵包含：
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$，每个头 $3d_{\text{model}}d_k$
- $h$ 个头的 $W_i^Q, W_i^K, W_i^V$ 合计 $3d_{\text{model}} \cdot (h \cdot d_k) = 3d_{\text{model}}^2$
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，参数 $d_{\text{model}}^2$
- **总计**：$4d_{\text{model}}^2$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 合并的 QKV 投影（效率更高）
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 一个线性层同时计算 Q, K, V
        qkv = self.W_qkv(Q).chunk(3, dim=-1)
        Q, K, V = [x.view(batch_size, -1, self.n_heads, self.d_k)
                     .transpose(1, 2) for x in qkv]
        
        # 计算注意力
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)
        
        # 拼接 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, -1, self.d_model)
        return self.W_o(attn_out)
```

---

## 4. 多查询注意力（Multi-Query Attention, MQA）

### 动机

在自回归解码中，MHA 需要为每个解码步加载完整的 K、V 缓存（KV Cache），内存带宽成为瓶颈。MQA 通过让所有头**共享同一个 K、V 投影**来大幅减少缓存大小。

### 定义

MQA（Shazeer, 2019）中：
- **Q**：每个头独立（$h$ 组）
- **K**：所有头共享（1 组）
- **V**：所有头共享（1 组）

$$\text{MQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)$

### 优点

- KV Cache 减少到原来的 $1/h$
- 解码速度显著提升（尤其在内存带宽受限时）
- 在推理效率上收益显著，质量下降有限

```python
class MultiQueryAttention(nn.Module):
    """多查询注意力 — 所有头共享 K、V"""
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q: 每个头独立; K, V: 所有头共享
        self.W_q = nn.Linear(d_model, d_model)       # 多头 Q
        self.W_k = nn.Linear(d_model, self.d_k)      # 单头 K
        self.W_v = nn.Linear(d_model, self.d_k)      # 单头 V
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Q 拆分为多头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K, V 保持单头，然后 unsqueeze 以便广播
        K = self.W_k(K).unsqueeze(1)  # [batch, 1, seq_k, d_k]
        V = self.W_v(V).unsqueeze(1)  # [batch, 1, seq_v, d_k]
        
        # 注意力计算（K, V 自动广播到所有头）
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)
        
        # 拼接 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, -1, self.d_model)
        return self.W_o(attn_out)
```

---

## 5. 分组查询注意力（Grouped Query Attention, GQA）

### 动机

MQA 的推理效率极高，但有时质量下降明显。GQA（Ainslie et al., 2023）是 MHA 和 MQA 的**折中方案**——将头分为 $g$ 组，每组共享一个 K、V。

### 定义

GQA 的配置通常用 $(h, g)$ 表示（$h$ 个 Q 头，$g$ 个 KV 组）：
- $g = h$ 时退化为 MHA
- $g = 1$ 时退化为 MQA
- $g$ 为中间值时（如 $g=4$ 或 $g=8$），在效率和质量之间取得平衡

```python
class GroupedQueryAttention(nn.Module):
    """分组查询注意力 — g 组共享 KV"""
    def __init__(self, d_model=512, n_heads=8, n_kv_groups=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.d_k = d_model // n_heads
        
        assert n_heads % n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"
        self.heads_per_group = n_heads // n_kv_groups
        
        self.W_q = nn.Linear(d_model, d_model)
        # K, V 只有 g 组，不是 h 组
        self.W_k = nn.Linear(d_model, n_kv_groups * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_groups * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Q: 拆为 h 个头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K, V: 拆为 g 组
        K = self.W_k(K).view(batch_size, -1, self.n_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_kv_groups, self.d_k).transpose(1, 2)
        
        # 将每组 KV 重复到组内的每个头
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, -1, self.d_model)
        return self.W_o(attn_out)
```

### GQA 的模型应用

| 模型 | MHA 配置 | GQA 配置 |
|------|---------|---------|
| LLaMA-2 70B | 64 头 | 8 KV 组（GQA） |
| LLaMA-3 8B | 32 头 | 8 KV 组（GQA） |
| LLaMA-3 70B | 64 头 | 8 KV 组（GQA） |
| Mistral 7B | 32 头 | 8 KV 组（GQA） |
| Gemma 7B | 16 头 | 16 KV 组（MHA，无 GQA） |

---

## 6. Flash Attention 核心思想

### 动机

标准 Attention 的实现在训练大模型时面临严重的**显存瓶颈**：

1. 需要计算并存储 $n \times n$ 的注意力分数矩阵到 HBM（高带宽显存）
2. 从 HBM 读取注意力分数做 softmax，再写回
3. 再从 HBM 读取 softmax 结果做加权求和

**问题**：对 HBM 的反复读写远慢于计算本身（计算是 compute-bound，但 Attention 是 memory-bound）。

### 核心思想

Flash Attention（Dao et al., 2022）通过**分块计算（tiling）**和**重计算（recomputation）**避免大注意力矩阵的显存存储：

1. **分块（Tiling）**：将 Q、K、V 切分成小块，在 SRAM（共享显存，速度快但容量小）中逐块计算注意力
2. **在线 softmax（Online Softmax）**：在不访问全部数据的情况下，通过维护局部统计量逐步计算全局 softmax
3. **重计算（Recomputation）**：反向传播时不从 HBM 读取注意力矩阵，而是在 SRAM 中重新计算

```python
# Flash Attention 的 PyTorch 原生等价（v2.0+）
def flash_attention(Q, K, V, mask=None):
    """
    PyTorch 2.0+ 原生支持的 scaled_dot_product_attention
    当输入为 float16/bfloat16 时会自动调用 Flash Attention 内核
    """
    # 一行代码等价于 hand-tuned Flash Attention
    return F.scaled_dot_product_attention(
        Q, K, V, 
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False  # 设为 True 自动生成因果掩码
    )

# 使用示例
Q = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')
K = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')
V = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device='cuda')

output = flash_attention(Q, K, V)  # 自动使用 Flash Attention 内核
```

### 性能提升

| 指标 | 标准 Attention | Flash Attention | 提升 |
|------|--------------|----------------|------|
| 显存占用 | $O(n^2)$ | $O(n)$ | 显著减少 |
| 训练速度 | 基准 | 2-4x 加速 | 2-4 倍 |
| 支持序列长度 | <= 2K（一般） | 可达 64K+ | 32 倍 |

---

## 注意力变体对比表格

| 特性 | MHA | MQA | GQA | Flash Attention |
|------|-----|-----|-----|-----------------|
| 提出时间 | 2017 | 2019 | 2023 | 2022 |
| Q 头数 | $h$ | $h$ | $h$ | 兼容任意 |
| K 头数 | $h$ | 1 | $g$ ($1 < g < h$) | 兼容任意 |
| V 头数 | $h$ | 1 | $g$ | 兼容任意 |
| KV Cache 大小 | $h d_k L$ | $d_k L$ | $g d_k L$ | N/A |
| 解码效率 | 基准 | 最高 | 中等 | N/A |
| 模型质量 | 最高 | 略低 | 接近 MHA | 与标准相同 |
| 显存节省 | 基准 | ~$h$ 倍 | ~$h/g$ 倍 | $O(n^2) \to O(n)$ |
| 代表模型 | BERT, T5 | PaLM | LLaMA-3, Mistral | GPT-4, LLaMA |

---

## 面试问答

### Q1: Multi-Head Attention 为什么有效？多个头学到了什么？

**A**: 多头注意力通过将模型拆分为多个子空间，让模型在不同维度上并行捕获不同类型的依赖关系。研究表明：

1. **不同头关注不同的语义/语法关系**：例如在机器翻译中，一些头关注语法依赖（主谓关系），另一些头关注位置邻近性，还有的头关注语义相似性。
2. **冗余与鲁棒性**：多个头之间有一定冗余，去掉个别头通常不会导致灾难性退化。
3. **特征多样性**：每个头的 $W_i^Q, W_i^K, W_i^V$ 投影到不同的子空间，使得不同头可以从不同角度衡量相似度。

**直观理解**：可以认为 MHA 是 Attention 的"集成学习（Ensemble）"——每个头是独立训练的注意力分类器，最后的拼接相当于投票/集成。

### Q2: MQA 和 GQA 为什么在质量下降不大的情况下还能提升推理速度？

**A**: 关键原因在于**内存带宽瓶颈**和**注意力头冗余**：

1. **内存带宽瓶颈**：在自回归解码中，每步生成都需要加载 KV Cache。KV Cache 的大小 = $2 \times h \times d_k \times L$（K 和 V 各一份）。当 $h$ 很大时（如 70B 模型有 64 个头），KV Cache 加载成为主要瓶颈。MQA 将 KV Cache 减少到 $1/h$，GQA 减少到 $h/g$，大大降低了内存带宽需求。

2. **注意力头冗余**：实验发现，MHA 中许多头的 K、V 投影学到的是相似的模式，共享 K、V 并不会显著损失表达能力。每个头仍然有独立的 Q 投影，可以关注不同的位置。

3. **经验证据**：LLaMA-3 70B 使用 GQA（64 头，8 组）在所有 benchmark 上匹配甚至超过了纯 MHA 同等规模模型的质量。

### Q3: Flash Attention 如何实现近乎免费的注意力计算？

**A**: Flash Attention 的关键洞察是：Attention 是 **memory-bound** 而非 compute-bound。它的优化策略分为三步：

**前向传播**：
1. **分块（Tiling）**：将大的 $Q, K, V$ 矩阵切分为小块，逐块加载到 SRAM 中计算
2. **在线 Softmax（Online Safe Softmax）**：无需一次性看到所有分数来计算 softmax，而是维护两个统计量 $m(x)$（逐行最大值）和 $\ell(x)$（逐行归一化因子），在遍历分块时增量更新
3. **融合内核（Kernel Fusion）**：将所有操作（矩阵乘法、mask、softmax、dropout、加权求和）融合到一个 CUDA kernel 中，避免在 HBM 和 SRAM 之间反复搬运数据

**反向传播**：
4. **重计算（Recomputation）**：不存储 $n \times n$ 注意力矩阵，而是在反向传播时用存储的统计量（$m, \ell$）在 SRAM 中快速重算

**效果**：HBM 读写从 $O(n^2)$ 降到 $O(n)$，虽然计算量不变（甚至略增），但 wall-clock 时间大幅降低。

### Q4: Self-Attention 的时间复杂度是 $O(n^2)$，有哪些主流的优化方向？

**A**: 主流优化方向包括：

1. **稀疏化（Sparsification）**
   - 滑动窗口注意力（Sliding Window）：每个位置只关注邻近 $w$ 个 token，复杂度 $O(nw)$
   - 膨胀滑动窗口（Dilated Sliding Window）：类似空洞卷积，扩大感受野
   - 全局 + 局部注意力：选部分 token 作为 global tokens（如 Longformer、BigBird）
   - 代表模型：Mistral（滑动窗口）、Longformer（全局+局部）

2. **线性化（Linearization）**
   - 将 softmax 注意力重写为 $\phi(Q)\phi(K)^\top V$ 形式（核方法）
   - 先计算 $\phi(K)^\top V$，复杂度 $O(n)$
   - 代表工作：Linear Transformer、Performer（FAVOR+）、RFA

3. **低秩近似（Low-Rank）**
   - 将注意力矩阵近似为低秩矩阵分解
   - 代表工作：Linformer

4. **硬件感知优化（Hardware-aware）**
   - Flash Attention 系列（目前最实用）
   - 不改变数学等价性，仅通过 IO-aware 算法加速

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019
3. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023
4. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
5. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
6. Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020
7. Kitaev et al., "Reformer: The Efficient Transformer", ICLR 2020
8. Child et al., "Generating Long Sequences with Sparse Transformers", 2019
