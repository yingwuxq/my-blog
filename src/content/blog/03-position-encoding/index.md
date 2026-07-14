---
title: "位置编码"
excerpt: "Sinusoidal、可学习、RoPE、ALiBi — 各种位置编码的原理与对比。"
date: 2024-06-15
category: "deep-learning"
tags: ["position-encoding", "transformer"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

Transformer 中的 Self-Attention 是**置换等变（Permutation Equivariant）**的——如果不加位置信息，改变输入 token 的顺序不会改变注意力输出。这意味着模型无法区分"我打你"和"你打我"的区别。位置编码（Position Encoding）就是为了解决这个问题。

本章系统梳理从经典到现代的位置编码技术。

---

## 1. 绝对位置编码（Absolute Position Encoding）

### Sinusoidal Position Encoding

Vaswani et al. (2017) 提出的原始方案，使用不同频率的正弦/余弦函数为每个位置生成固定向量：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $pos$ 是位置索引，$i$ 是维度索引，$d_{\text{model}}$ 是模型维度。

```python
import torch
import math
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(seq_len: int, d_model: int):
    """
    生成 Sinusoidal 位置编码
    
    Args:
        seq_len: 序列最大长度
        d_model: 模型维度
    Returns:
        pe: [seq_len, d_model]  位置编码矩阵
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    
    # 计算频率：10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
    
    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe  # [seq_len, d_model]

# 可视化不同维度随位置的变化
pe = sinusoidal_positional_encoding(100, 512)
# pe[:, 0] 最低频，pe[:, 100] 较高频
print(f"Position Encoding Shape: {pe.shape}")
```

**优点**：
- 不需要学习参数（固定编码）
- 可以外推到训练时未见过的序列长度
- 每个位置有唯一的编码向量

**缺点**：
- 绝对位置信息对 Transformer 的意义有限（模型更需要相对位置关系）
- 外推性能不如专门设计的相对编码

### Learned Position Encoding

BERT 和 GPT 等模型使用的可学习位置编码：

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        # 可学习的嵌入表
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

**问题**：最大序列长度在训练时固定，不能外推超过训练长度的序列。

---

## 2. 相对位置编码（Relative Position Encoding）

### 核心思想

Self-Attention 的权重应该取决于两个位置之间的**相对距离**而非绝对位置。Shaw et al. (2018) 在计算注意力分数时引入了可学习的相对位置偏置：

$$a_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^\top}{\sqrt{d_k}}$$

其中 $a_{ij}^K$ 是位置 $i$ 和 $j$ 之间相对距离的可学习嵌入。

T5 使用了一种简化的相对位置偏置：将注意力分数加上一个标量偏置 $b_{pos(i,j)}$，偏置值只依赖于 $i$ 和 $j$ 的相对偏移，且不同层共享：

$$a_{ij} = \frac{x_i W^Q (x_j W^K)^\top}{\sqrt{d_k}} + b_{pos(i,j)}$$

---

## 3. RoPE（Rotary Position Embedding）

### 概述

RoPE（Su et al., 2021）是目前最流行的位置编码方案之一，被 LLaMA、Mistral、Qwen 等主流模型采用。其核心思想是将位置信息**通过旋转矩阵施加到 Query 和 Key 上**。

### 数学原理

对于位置 $m$ 处的 token，其第 $i$ 维的查询 $q_m^{(i)}$ 和键 $k_n^{(i)}$ 通过旋转矩阵变换：

$$f_{\{q,k\}}(x_m, m) = R_{\Theta,m} \cdot W_{\{q,k\}} x_m$$

其中 $R_{\Theta,m}$ 是块对角旋转矩阵：

$$R_{\Theta,m} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

$\theta_i = base^{-2(i-1)/d}$，其中 $base$ 通常取 10000（LLaMA 使用 500000 以获得更好的长序列外推能力）。

旋转后的注意力分数仅依赖于相对位置：

$$q_m^\top k_n = (R_{\Theta,m} W^Q x_m)^\top (R_{\Theta,n} W^K x_n) = (W^Q x_m)^\top R_{\Theta,m-n} (W^K x_n)$$

### 实现

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE — Rotary Position Embedding
    参考 LLaMA 实现
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # 计算频率 theta_i
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        x: [batch, seq_len, n_heads, d_k] 或 [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # 生成位置索引 [offset, offset+1, ..., offset+seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        t = t + offset
        
        # 计算角度: [seq_len, d_k/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # 拼接 cos 和 sin: [seq_len, d_k]
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        
        return cos, sin

def rotate_half(x):
    """旋转一半维度: [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """将 RoPE 应用到 Q 和 K 上"""
    # q, k: [batch, seq_len, n_heads, d_k]
    # cos, sin: [seq_len, d_k]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_k]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# 使用示例
dim = 128
rope = RotaryPositionalEmbedding(dim=dim, base=10000.0)
q = torch.randn(2, 10, 8, dim // 8)  # [batch, seq, heads, d_k]
k = torch.randn(2, 10, 8, dim // 8)
cos, sin = rope(q)
q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
print(f"RoPE applied: q shape {q_rope.shape}, k shape {k_rope.shape}")
```

### RoPE 的特性

| 特性 | 说明 |
|------|------|
| 相对位置 | Attention 分数仅依赖于相对位置差 |
| 远程衰减 | 距离越远的 token 间最大注意力分数越小（类似 Sinusoidal） |
| 可外推 | 通过修改 base 或插值，可扩展到更长序列 |
| 兼容性 | 可以和 Flash Attention、GQA 等无缝结合 |
| 无额外参数 | 不需要额外的位置嵌入表 |

### RoPE 的外推方法

- **NTK-aware Scaling**：通过调整 $base$ 值（如增加到 500000）来扩展有效上下文
- **YaRN**：对每个维度的频率应用不同的缩放因子
- **线性插值（PI）**：将位置索引缩放 $s$ 倍（如 4x 插值）

---

## 4. ALiBi（Attention Linear Biases）

### 概述

ALiBi（Press et al., 2022）提出了一种极其简洁的位置编码方案——**不向 Embedding 添加位置信息**，而是在注意力分数上施加线性偏置：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + m \cdot B\right)V$$

其中 $B$ 是一个固定的偏置矩阵，$B_{i,j} = -|i-j|$，$m$ 是一个头相关的斜率（head-specific slope）：

$$m = 2^{-2^{(\log_2(h+3)) - (i+1)}}$$

### 实现

```python
class AlibiAttention(nn.Module):
    """ALiBi 位置编码（仅需在注意力分数上添加偏置）"""
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        # 计算每个头的斜率
        self.slopes = self._compute_alibi_slopes(n_heads)
        
    def _compute_alibi_slopes(self, n_heads):
        def get_slopes(n):
            def _get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                return [start * 2 ** (-i) for i in range(n)]
            
            if math.log2(n).is_integer():
                return _get_slopes_power_of_2(n)
            else:
                closest_power = 2 ** math.floor(math.log2(n))
                slopes = _get_slopes_power_of_2(closest_power)
                extra = n - closest_power
                slopes += [slopes[-1] * 2 ** (i + 1) for i in range(extra)]
                return slopes
        
        return torch.tensor(get_slopes(n_heads))
    
    def forward(self, scores, seq_len_q, seq_len_k):
        """
        scores: [batch, heads, seq_q, seq_k] (缩放后的注意力分数)
        返回添加了 ALiBi 偏置的分数
        """
        # 构建距离矩阵 [seq_q, seq_k]
        pos_i = torch.arange(seq_len_q).unsqueeze(1)
        pos_j = torch.arange(seq_len_k).unsqueeze(0)
        distance = -(pos_i - pos_j).abs()  # [seq_q, seq_k]
        
        # 对每个头乘以对应的斜率
        alibi_bias = distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        # alibi_bias: [heads, seq_q, seq_k]
        
        return scores + alibi_bias.unsqueeze(0)  # [batch, heads, seq_q, seq_k]
```

### ALiBi 特性

- **零额外参数**：不需要学习或存储位置编码
- **天然支持外推**：训练时短序列，推理时长序列（测试到 2x 以上仍有效）
- **简单高效**：只需在注意力分数上加一个线性偏置

---

## 5. NoPE（No Position Encoding）

### 概述

**No Position Encoding**（NoPE, Haviv et al., 2022）发现因果语言模型（Causal LM）在**完全没有位置编码**的情况下，通过因果掩码（Causal Mask）本身就能隐式地编码位置信息：

1. 因果掩码的非对称结构提供了绝对位置信息（第一个 token 只能看 1 个位置，第 n 个 token 能看 n 个位置）
2. 不同的 token 位置看到的序列长度不同，因此其表示天然包含位置信息

虽然 NoPE 在小规模实验中可行，但在大规模场景下仍未被主流采用。

---

## 6. 对比表格

| 特性 | Sinusoidal | Learned | Relative (T5) | RoPE | ALiBi | NoPE |
|------|-----------|---------|---------------|------|-------|------|
| 提出时间 | 2017 | 2018 | 2018 | 2021 | 2022 | 2022 |
| 参数量 | 0 | $O(L_{\max}d)$ | $O(L_{\max})$ | 0 | 0 | 0 |
| 相对位置 | 隐式 | 隐式 | 显式 | 显式 | 显式 | 仅因果 |
| 外推能力 | 有限 | 无 | 有限 | 好（需NTK） | 优秀 | 有限 |
| 实现复杂度 | 低 | 低 | 中 | 中 | 极低 | 无 |
| 训练效率 | 高 | 高 | 中 | 高 | 最高 | 最高 |
| 与 Flash Attn 兼容 | 直接 | 直接 | 需修改 | 需修改 | 需修改 | 直接 |
| 使用模型 | Transformer | BERT, GPT-2 | T5 | LLaMA, Mistral | BLOOM | 少数实验 |

---

## 面试问答

### Q1: RoPE 为什么能编码相对位置？

**A**: RoPE 通过旋转矩阵将位置信息编码到 Query 和 Key 中，使得 Attention 分数 $q_m^\top k_n$ 仅依赖于相对位置 $m-n$。

数学推导：设 $q_m = R_m \cdot q$，$k_n = R_n \cdot k$，其中 $R_m$ 是旋转矩阵。则：

$$q_m^\top k_n = (R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

因为旋转矩阵满足 $R_m^\top R_n = R_{n-m}$（旋转矩阵的转置等于逆矩阵，且旋转操作可加：$R_{-m} R_n = R_{n-m}$）。因此注意力分数仅依赖于 $(n-m)$，即相对位置。

### Q2: ALiBi 相比 RoPE 有什么优缺点？

**A**:

**ALiBi 优点**：
1. **实现极简**：不需要修改 Q、K 的计算，仅在注意力分数上加一个固定的线性偏置
2. **外推能力优秀**：在 2x-3x 的外推场景下表现优异，无需微调
3. **零参数量**：不需要嵌入表或可学习参数

**ALiBi 缺点**：
1. **线性偏置过于简单**：在所有维度上使用相同的偏置模式，表达能力不如 RoPE 的旋转编码丰富
2. **无法通过 fine-tuning 提升**：偏置是固定的，无法通过后训练适应新的分布
3. **长距离外推不如 RoPE + NTK 扩展**：在 10x+ 的极长序列外推中，RoPE 结合插值方法更有优势

**RoPE 优点**：
1. **对于长上下文的 adaptation 更友好**：可以通过 Position Interpolation、NTK-aware scaling 等方法扩展到极长序列
2. **复杂的频率模式**：不同维度使用不同频率，表达能力更强

**工程选择**：目前 RoPE 是主流选择（LLaMA、Mistral、Qwen 等），ALiBi 主要用于 BLOOM 等模型。

### Q3: 什么是位置编码的外推（Extrapolation）？为什么 Sinusoidal 的外推能力有限？

**A**: 位置编码的**外推**是指模型在训练时见过的最大序列长度 $L_{\text{train}}$ 上训练，但在推理时能够处理更长的序列 $L_{\text{test}} > L_{\text{train}}$。

**Sinusoidal 外推能力有限的原因**：

虽然 Sinusoidal 在数学上可以产生任意长度的编码向量，但实际中模型在外推时的表现下降很快。原因在于：

1. **注意力分布偏移**：模型在训练时学会的注意力模式是针对 $L_{\text{train}}$ 范围内的位置距离的。对于超出训练范围的位置，模型未见过的相对距离 $> L_{\text{train}}$ 会导致注意力分布出现未预期的模式。
2. **频率学习不均衡**：低频维度（高 $i$）在训练序列长度内仅变化了几个周期，模型无法学到这些维度的完整模式。当外推到更长序列时，这些维度会出现未预期数值。

**不同方案的外推能力排序**：
ALiBi > RoPE (NTK) > Sinusoidal > Learned

### Q4: 为什么 LLaMA 将 RoPE 的 base 从 10000 改到 500000？

**A**: 更大的 base 值意味着 $\theta_i = base^{-2i/d}$ 整体变小，频率降低。具体效果：

1. **降低最低频维度（最大周期）**：base=10000 时，最低频维度的周期约为 $2\pi / 10^{0} = 2\pi$；base=500000 时，最低频维度的周期约为 $2\pi \cdot 50 = 314\pi$ 个 token。更大的周期意味着长距离位置仍然能通过旋转角度区分。
2. **提升长距离位置的分辨率**：在长序列场景下，两个距离较远的位置如果使用高频旋转向量，它们的旋转角度差可能超过 $2\pi$ 导致混叠（aliasing）。降低频率使角度差始终保持在 $(0, 2\pi)$ 范围内，保留了相对位置的分辨率。
3. **缓解外推性能下降**：更大的 base 在训练时使用较少周期的编码，使模型更容易外推到未见过的长序列。

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", 2017 (Sinusoidal PE)
2. Shaw et al., "Self-Attention with Relative Position Representations", NAACL 2018
3. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
4. Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", NeurIPS 2022
5. Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information", EMNLP 2022
6. Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation", 2023
7. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", 2023
8. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023
