---
title: "损失函数"
excerpt: "交叉熵、对比损失、CTC 损失等，覆盖分类、生成、序列任务的损失设计。"
date: 2024-06-15
category: "deep-learning"
tags: ["loss-function", "deep-learning"]
author: "yingwu"
featured: false
draft: false
---

## 概述

损失函数（Loss Function / Objective Function）度量模型预测与真实目标之间的差距，是模型优化的直接信号。本章覆盖语言模型、对比学习等领域的关键损失函数，包含完整的数学推导和 PyTorch 实现。

---

## 1. Cross-Entropy Loss（交叉熵损失）

### 定义

交叉熵衡量两个概率分布之间的差异，是多分类任务的标准损失函数：

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

其中：
- $y$ 是真实标签的 one-hot 向量（或软标签）
- $\hat{y} = \text{softmax}(z)$ 是模型预测的概率分布
- $C$ 是类别数

对于单个样本，真实类别为 $c$ 时简化为：

$$\mathcal{L}_{\text{CE}} = -\log(\hat{y}_c) = -\log\left(\frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}\right)$$

### 梯度推导

对 logits $z_k$ 的梯度——这是反向传播中的关键公式：

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = \hat{y}_k - y_k = \text{softmax}(z)_k - y_k$$

这个梯度有清晰的物理意义：**预测概率与真实标签的差异**。当 $\hat{y}_k = y_k$ 时梯度为 0（收敛）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    手动实现 Cross-Entropy Loss（与 F.cross_entropy 等价）
    
    Args:
        logits: [batch, num_classes] 未经 softmax 的原始输出
        targets: [batch] 真实类别索引（0-indexed）
    Returns:
        标量损失
    """
    # 1. Log-Softmax 合并计算（数值稳定）
    # log_softmax = logits - logsumexp(logits)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 2. 收集目标类别对应的 log-prob
    batch_size = logits.size(0)
    loss = -log_probs[torch.arange(batch_size), targets].mean()
    
    return loss

# 验证
logits = torch.randn(4, 10)  # [batch=4, classes=10]
targets = torch.tensor([3, 7, 1, 9])

manual_loss = cross_entropy_loss_manual(logits, targets)
torch_loss = F.cross_entropy(logits, targets)

print(f"Manual CE: {manual_loss.item():.6f}")
print(f"PyTorch CE: {torch_loss.item():.6f}")
print(f"Match: {torch.allclose(manual_loss, torch_loss)}")
```

### 数值稳定性

直接计算 $\mathcal{L} = -\log\left(e^{z_c} / \sum e^{z_j}\right)$ 可能溢出（$e^{z_j}$ 可能极大）。推荐的稳定实现：

```python
def stable_cross_entropy(logits, targets):
    """数值稳定的交叉熵实现"""
    # 减去 max logits 防止指数爆炸
    # softmax(x) = softmax(x - max(x))
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
    
    # log-sum-exp 稳定计算
    log_sum_exp = torch.log(torch.exp(logits_stable).sum(dim=-1))
    
    # 目标类别的 logit
    batch_size = logits.size(0)
    target_logits = logits_stable[torch.arange(batch_size), targets]
    
    loss = -(target_logits - log_sum_exp).mean()
    return loss
```

### 在语言模型中的应用

语言模型的 Cross-Entropy 是在词汇表上的分类问题：

```python
class LanguageModelLoss(nn.Module):
    """语言模型的交叉熵损失（含掩码）"""
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]  (-100 表示忽略该位置)
        """
        # 展平为 [batch*seq_len, vocab_size]
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        return F.cross_entropy(logits_flat, labels_flat, ignore_index=self.ignore_index)
```

**Perplexity（困惑度）**：语言模型评估的核心指标，直接由 Cross-Entropy 导出：

$$\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$$

---

## 2. Masked Language Model Loss（MLM 损失）

### 定义

BERT 使用的掩码语言模型损失，是 Cross-Entropy 的特例——只在被掩码的位置计算损失：

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i | x_{\setminus M})$$

其中 $M$ 是被掩码位置集合，$|M|$ 是掩码位置总数，$x_{\setminus M}$ 是未掩码的上下文。

```python
class MaskedLMLoss(nn.Module):
    """BERT MLM 损失 —— 只在掩码位置计算"""
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]  
          - 正常 token ID: 该位置在 MLM 中使用
          - -100: 该位置不计算损失
        """
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        # 只对 labels != -100 的位置计算损失
        loss = loss_fct(
            logits.view(-1, self.vocab_size),
            labels.view(-1)
        )
        return loss

# 使用示例
batch_size, seq_len, vocab_size = 4, 128, 30522
logits = torch.randn(batch_size, seq_len, vocab_size)

# 15% 的 token 被掩码
labels = torch.randint(0, vocab_size, (batch_size, seq_len))
mask = torch.rand(batch_size, seq_len) < 0.15
labels[~mask] = -100  # 非掩码位置不计算损失

mlm_loss = MaskedLMLoss(vocab_size)
loss = mlm_loss(logits, labels)
print(f"MLM Loss: {loss.item():.4f}")

# 验证只计算了掩码位置
n_masked = (labels != -100).sum().item()
print(f"Total masked positions: {n_masked} / {batch_size * seq_len}")
```

### BERT 的 80-10-10 策略

BERT 预训练时，对被选中的 15% token 进行以下替换：
- 80% 替换为 `[MASK]`（用于模型修复掩码）
- 10% 替换为随机 token（引入噪声，增强鲁棒性）
- 10% 保持不变（迫使模型关注上下文，而非单纯记忆）

```python
def bert_mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    """BERT 风格的掩码策略"""
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_prob)
    
    # 特殊 token 不掩码
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 非掩码位置标签设为 -100
    
    # 80% MASK, 10% random, 10% original
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels
```

---

## 3. Contrastive Loss（对比损失）

### InfoNCE Loss（CLIP 使用）

对比损失将正样本对拉近、负样本对推远。CLIP 使用的 InfoNCE Loss：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i^a, z_i^p) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i^a, z_j^p) / \tau)}$$

其中 $\text{sim}$ 通常是余弦相似度，$\tau$ 是温度参数。

```python
class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent 对比损失
    
    在 CLIP 中使用：图像和文本嵌入的对比学习
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """
        z_a, z_p: [batch, dim]  两个模态/视角的嵌入
        """
        batch_size = z_a.size(0)
        
        # 归一化到单位向量
        z_a = F.normalize(z_a, dim=-1)
        z_p = F.normalize(z_p, dim=-1)
        
        # 计算相似度矩阵 [batch, batch]
        # sim[i][j] = z_a[i] · z_p[j]
        logits = torch.matmul(z_a, z_p.T) / self.temperature
        
        # 对角线为正样本对（i 与 i 匹配）
        labels = torch.arange(batch_size, device=z_a.device)
        
        # 对称损失: image→text 和 text→image 的交叉熵
        loss_a = F.cross_entropy(logits, labels)      # 行方向: 每行是对应的文本
        loss_p = F.cross_entropy(logits.T, labels)    # 列方向: 每列是对应的文本
        
        return (loss_a + loss_p) / 2.0

# CLIP 风格训练示例
batch_size, dim = 32, 512
image_embeds = torch.randn(batch_size, dim)  # 图像编码器输出
text_embeds = torch.randn(batch_size, dim)   # 文本编码器输出

contrastive_loss = InfoNCELoss(temperature=0.07)
loss = contrastive_loss(image_embeds, text_embeds)
print(f"Contrastive Loss: {loss.item():.4f}")
```

### 温度参数 $\tau$ 的作用

温度控制对难负样本的关注程度：
- **低温度**（如 0.07）：放大 logits 差异，更关注难负样本（hard negatives）
- **高温度**（如 1.0）：均匀化注意力，所有负样本被平等对待

---

## 4. KL Divergence Loss（KL 散度损失）

### 定义

KL 散度衡量两个概率分布 $P$ 和 $Q$ 之间的差异：

$$D_{\text{KL}}(P \parallel Q) = \sum_i P(i) \left( \log P(i) - \log Q(i) \right) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

**注意**：$D_{\text{KL}}$ 是不对称的——$D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$。

### 在 LLM 中的应用（Distillation + RLHF）

#### 知识蒸馏
学生模型 $Q$ 模仿教师模型 $P$ 的输出分布：

```python
def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    知识蒸馏损失（Hinton et al., 2015）
    
    使用较高的温度软化概率分布，让学生学到教师的知识
    """
    # 带温度的 softmax
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL(P_teacher || Q_student)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
    
    # 温度缩放：乘以 T^2 保持梯度尺度
    return kl_loss * temperature * temperature
```

#### RLHF 中的 KL Penalty

在 RLHF 中，KL 散度作为**正则化项**，防止模型在 RL 微调中偏离初始 SFT 模型太远：

$$\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{x \sim \pi_\theta} [R(x)] + \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{SFT}})$$

```python
def rlhf_kl_penalty(
    log_probs: torch.Tensor,      # 当前策略 log π_θ
    ref_log_probs: torch.Tensor,  # 参考策略 log π_ref
    kl_coef: float = 0.04
) -> torch.Tensor:
    """
    RLHF 中的 KL 惩罚
    
    几种常见变体:
    - 'kl_penalty': β * KL(π_θ || π_ref)  （标准）
    - 'adaptive_kl': 动态调整 β
    """
    # KL(π_θ || π_ref) = ∑ π_θ * (log π_θ - log π_ref)
    # 在 PPO 中简化为: log(π_θ/π_ref) 的期望
    kl_divergence = torch.exp(log_probs) * (log_probs - ref_log_probs)
    return kl_coef * kl_divergence.mean()
```

### Cross-Entropy 与 KL Divergence 的关系

当使用 one-hot 标签时，Cross-Entropy 等价于 KL Divergence 加一个常数：

$$\mathcal{L}_{\text{CE}} = D_{\text{KL}}(y_{\text{true}} \parallel \hat{y}) + H(y_{\text{true}})$$

由于 $H(y_{\text{true}}) = 0$（one-hot 的熵为 0），两者在分类问题中等价。

---

## 损失函数对比表格

| 损失函数 | 公式 | 输入 | 输出范围 | 主要应用 | 特性 |
|----------|------|------|---------|---------|------|
| Cross-Entropy | $-\sum y \log \hat{y}$ | logits + 标签 | $[0, \infty)$ | 分类、语言模型 | 标准分类损失 |
| MLM Loss | $-\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid \cdot)$ | logits + 掩码标签 | $[0, \infty)$ | BERT 预训练 | 仅在掩码位置计算 |
| InfoNCE（对比） | $-\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ | 嵌入对 | $(-\infty, \infty)$ | CLIP、SimCLR | 拉近正对、推远负对 |
| KL Divergence | $\sum P \log(P/Q)$ | 两个分布 | $[0, \infty)$ | 蒸馏、RLHF | 不对称性需要注意 |

### 梯度对比

| 损失函数 | 梯度形式 | 梯度特性 |
|----------|---------|---------|
| Cross-Entropy | $\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$ | 简单、稳定 |
| InfoNCE | 正样本对吸引 + 负样本对排斥 | 难负样本影响大 |
| KL Divergence | $\frac{\partial D_{KL}}{\partial z_q} = q - p$（对 $\log q$） | 与 CE 类似 |

---

## 面试问答

### Q1: 为什么语言模型使用 Perplexity 而不是直接使用 Cross-Entropy Loss 评估？

**A**: Perplexity（困惑度）和 Cross-Entropy 本质上是等价的——$\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$，但 Perplexity 有更直观的解释：

1. **可解释性**：Perplexity 可以理解为模型在预测下一个 token 时的"有效平均候选数"。PPL = 10 意味着模型平均在 10 个候选 token 中不确定选哪一个。这比 Cross-Entropy 的数值（如 2.3）更容易理解。

2. **跨模型可比**：由于 PPL 是指数变换，它在数值上更容易感知差异。PPL 从 20 到 10 意味着不确定性降低了一半。

3. **信息论解释**：Perplexity 与 zipf 定律相关——人类语言的自然 perplexity（基于 n-gram）约为 100-200，好的 LLM 可以将 PPL 降到 10 以下。

**注意**：Perplexity 只在**相同分词器**的模型之间可比。不同 tokenization 会显著影响 PPL 数值。

### Q2: InfoNCE Loss 中的温度参数 $\tau$ 如何影响学习？

**A**: 温度 $\tau$ 控制对比学习对难负样本的关注程度：

- **$\tau$ 很小（如 0.07）**：Logits 被放大，softmax 分布变得更尖锐。正样本的概率很高，但梯度会主要关注**最相似的负样本**（hard negatives）。这使模型更善于区分相似的样本，但也更可能导致表示空间崩塌（collapse）或训练不稳定。

- **$\tau$ 很大（如 1.0）**：Logits 被缩小，softmax 分布变得更平滑。所有负样本被几乎平等地对待。梯度较为均匀，训练更稳定，但区分能力较弱。

- **$\tau$ 的直觉**：可以理解为"相似度的缩放因子"。当 $\tau$ 到 0 时，InfoNCE 退化为 hardmax（只关注最近邻负样本）；当 $\tau \to \infty$ 时，梯度消失。

**经验值**：CLIP 使用 0.07，SimCLR 使用 0.5。理论分析表明最优 $\tau$ 与 batch size 正相关。

### Q3: 知识蒸馏中为什么使用 KL Divergence 而非 Cross-Entropy？

**A**: 知识蒸馏（Knowledge Distillation）使用 KL 散度而非 Cross-Entropy 有两个原因：

1. **概率对概率**：蒸馏是两个**概率分布**之间的知识迁移——教师模型的 softmax 输出是软标签（soft label），学生模型需要匹配这个分布而非简单的 hard label。KL 散度是衡量两个分布差异的自然选择。

2. **不对称性的意义**：$D_{\text{KL}}(P_{\text{teacher}} \parallel Q_{\text{student}})$ 表示"在教师分布下，学生与教师的平均差异"。这意味着：
   - 当教师对某个类别非常确定时（$P$ 很大），学生必须准确匹配
   - 当教师不确定时（$P$ 很小），学生可以不精确匹配
   
   这种不对称性符合教育场景——重点知识必须学会，边缘信息可以忽略。如果使用对称的 JS 散度，这个特性会丢失。

**注意**：在分类任务中，Cross-Entropy 和 KL 散度的数学表达式非常相似——区别在于 KL 需要知道教师分布 $P$，而 CE 直接使用 one-hot 标签。

### Q4: MLM Loss 为什么使用 80-10-10 的策略？

**A**: BERT 的 80-10-10 策略是为了解决**预训练-微调不一致**（Pretrain-Finetune Discrepancy）问题：

- 如果**100% 将被选中的 token 替换为 [MASK]**：模型只在看到 [MASK] 时做预测。但在微调阶段，模型**永远不会看到 [MASK]**（下游任务没有掩码）。这种不一致会损害微调效果。

- **10% 保持不变**：迫使模型关注上下文信息来正确预测 token，而不是单纯依赖 [MASK] 标记。这让模型学会利用上下文表达。

- **10% 替换为随机 token**：引入噪声，让模型学会纠正错误上下文。这增强了模型的鲁棒性——即使输入中有错误 token，模型仍能给出合理的预测。

- **80% 替换为 [MASK]**：保持足够的 [MASK] 信号，确保模型学习双向上下文表示。

实验表明，去掉 10% 的随机令牌或 10% 的不变令牌都会导致微调性能下降。

---

## 参考文献

1. Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2015 (KL Distillation)
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019 (MLM)
3. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 (CLIP / InfoNCE)
4. van den Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018 (InfoNCE)
5. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020 (SimCLR)
6. Schulman et al., "Proximal Policy Optimization Algorithms", 2017 (RLHF + KL)
7. Ziegler et al., "Fine-Tuning Language Models from Human Preferences", 2019 (RLHF)
