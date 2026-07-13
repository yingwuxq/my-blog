---
title: "优化器"
excerpt: "Adam、AdamW、SGD 及其变体的原理、公式与训练稳定性分析。"
date: 2024-06-15
category: "deep-learning"
tags: ["optimizer", "deep-learning"]
author: "yingwu"
featured: false
draft: false
---

## 概述

优化器（Optimizer）是深度学习训练的核心引擎，负责根据梯度更新模型参数以最小化损失函数。本章从 SGD 发展到 AdamW 和 LAMB，覆盖主流优化器的算法原理、PyTorch 实现和使用场景。

---

## 1. SGD（随机梯度下降）

### 基础 SGD

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

其中 $\eta$ 是学习率，$\nabla \mathcal{L}$ 是 loss 对参数的梯度。

```python
import torch
import torch.nn as nn

def sgd_step(params, grads, lr=0.01):
    """单步 SGD 更新"""
    with torch.no_grad():
        for param, grad in zip(params, grads):
            param -= lr * grad

# PyTorch 内置
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### SGD + Momentum

动量（Momentum）在梯度更新中加入"惯性"，减少震荡并加速收敛：

$$v_{t+1} = \mu v_t + \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

其中 $\mu$ 是动量系数（通常 0.9），$v$ 是速度（velocity）。

```python
class SGDMomentum:
    """
    SGD + Momentum 手动实现
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        with torch.no_grad():
            for param, vel, grad in zip(self.params, self.velocities, grads):
                vel.mul_(self.momentum).add_(grad)  # v = μv + g
                param.sub_(self.lr * vel)  # θ -= ηv
                
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# 使用
model = nn.Linear(10, 1)
sgd_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Nesterov Momentum**（NAG）是动量的改进版，先"看看前方"再更新：

$$v_{t+1} = \mu v_t + \nabla \mathcal{L}(\theta_t - \eta \mu v_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

```python
# Nesterov Accelerated Gradient
optimizer_nag = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## 2. AdaGrad（自适应梯度）

### 原理

AdaGrad（Duchi et al., 2011）为每个参数分配**独立的自适应学习率**，频繁更新的参数学习率衰减更快，稀疏更新的参数保持较大学习率：

$$G_{t+1} = G_t + (\nabla \mathcal{L}(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \nabla \mathcal{L}(\theta_t)$$

其中 $G_t$ 是历史梯度平方的累积和，$\epsilon$ 防止除零。

```python
class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.G = [torch.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        with torch.no_grad():
            for param, G, grad in zip(self.params, self.G, grads):
                G.add_(grad.pow(2))  # G += g^2
                param.sub_(self.lr * grad / (G.sqrt() + self.eps))
```

**问题**：$G_t$ 单调递增，学习率持续衰减到接近 0，导致训练提前停止。

---

## 3. RMSProp

### 原理

RMSProp（Hinton, 2012）用**指数移动平均（EMA）**代替 AdaGrad 的累积和，解决了学习率持续衰减问题：

$$v_{t+1} = \beta v_t + (1-\beta) (\nabla \mathcal{L}(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1} + \epsilon}} \nabla \mathcal{L}(\theta_t)$$

其中 $\beta$ 通常取 0.9（近 10 步的梯度平方的加权平均）。

```python
class RMSProp:
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = [torch.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        with torch.no_grad():
            for param, v, grad in zip(self.params, self.v, grads):
                v.mul_(self.beta).addcmul_(grad, grad, value=1-self.beta)
                param.sub_(self.lr * grad / (v.sqrt() + self.eps))
```

---

## 4. Adam（Adaptive Moment Estimation）

### 算法完整流程

Adam（Kingma & Ba, 2015）结合了 Momentum（一阶矩估计）和 RMSProp（二阶矩估计），是目前最广泛使用的优化器。

**算法**：对于每个参数 $\theta$ 在时间步 $t$：

1. **计算梯度**：$g_t = \nabla \mathcal{L}(\theta_{t-1})$
2. **更新有偏一阶矩估计**（动量）：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
3. **更新有偏二阶矩估计**（自适应学习率）：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4. **偏差修正**：$\hat{m}_t = \dfrac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \dfrac{v_t}{1-\beta_2^t}$
5. **参数更新**：$\theta_t = \theta_{t-1} - \eta \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

其中 $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ 是默认值。

### 偏差修正（Bias Correction）的原因

在训练初期，$m_t$ 和 $v_t$ 被初始化为 0，导致它们被严重低估。偏差修正通过除以 $1-\beta^t$ 来消除这种初始化偏差：

- $t=1$ 时：$\hat{m}_1 = m_1 / (1-0.9) = m_1 / 0.1$，将一阶矩放大 10 倍
- $t \to \infty$ 时：$1-\beta^t \to 1$，修正消失

```python
class Adam:
    """
    Adam 优化器完整实现
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 时间步计数
        self.m = [torch.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]  # 二阶矩
        
    def step(self, grads):
        self.t += 1
        with torch.no_grad():
            for param, m, v, grad in zip(self.params, self.m, self.v, grads):
                # 1. 更新有偏矩估计
                m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # 2. 偏差修正
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # 3. 参数更新
                param.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# 验证与 PyTorch 一致性
model = nn.Linear(5, 2)
x = torch.randn(4, 5)
y = torch.randn(4, 2)

# 手动 Adam
manual_adam = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 训练一步
out = model(x)
loss = loss_fn(out, y)
loss.backward()

grads = [p.grad.clone() for p in model.parameters()]
manual_adam.step(grads)

params_after_manual = [p.clone().detach() for p in model.parameters()]
print("Manual Adam step done")
```

---

## 5. AdamW（权重解耦）

### 动机

原始 Adam 在实现权重衰减（Weight Decay）时，将其合并到了梯度中：
$$\nabla \mathcal{L}' = \nabla \mathcal{L} + \lambda \theta$$

但这与 Adam 的自适应学习率交互时会产生问题——权重衰减的效果被自适应学习率缩放，导致不同参数的衰减程度不一致。

### AdamW 的改进

AdamW（Loshchilov & Hutter, 2019）将权重衰减从梯度更新中**解耦**，直接在参数更新后独立执行：

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

```python
class AdamW:
    """
    AdamW — Adam with Decoupled Weight Decay
    
    将权重衰减与自适应梯度更新解耦
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        self.t += 1
        with torch.no_grad():
            for param, m, v, grad in zip(self.params, self.m, self.v, grads):
                # 标准的 Adam 动量更新
                m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # 解耦的权重衰减：直接在参数上施加衰减
                # PyTorch 官方实现方式
                param.mul_(1 - self.lr * self.weight_decay)
                param.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))

# PyTorch 内置（实现稍有不同但等价）
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### AdamW vs Adam

| 特性 | Adam | AdamW |
|------|------|-------|
| 权重衰减实现 | 合并到梯度中 | 解耦，独立更新 |
| 自适应 LR 与 WD 的交互 | WD 被自适应 LR 缩放 | WD 不受 LR 缩放影响 |
| 泛化性能（SOTA） | 通常较差 | 更好 |
| 超参数敏感度 | 较高 | 较低 |
| 使用模型 | 早期模型 | GPT, LLaMA, BERT 等 |

---

## 6. LAMB（Layer-wise Adaptive Moments）

### 动机

在大 batch size 训练（如 64K）时，Adam 的学习率需要大幅降低，导致训练速度下降。LAMB（You et al., 2020）通过**逐层自适应学习率**解决了这个问题。

### 核心思想

LAMB = Adam + 逐层学习率缩放：

$$r_t = \frac{||\theta_{t-1}||_2}{||\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)||_2}$$

$$\theta_t = \theta_{t-1} - \eta \cdot \underbrace{\max(\min(r_t, r_{\max}), r_{\min})}_{\text{信任比率}} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

信任比率（Trust Ratio）确保更新后的参数范数与原始参数范数在同一量级。

```python
class LAMB:
    """
    LAMB — Layer-wise Adaptive Moments for Batch training
    
    关键：逐层计算信任比率，使得大 batch 训练可以使用大学习率
    """
    def __init__(self, params, lr=0.004, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        
    def step(self, grads):
        self.t += 1
        with torch.no_grad():
            for param, m, v, grad in zip(self.params, self.m, self.v, grads):
                m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # 更新方向
                update = m_hat / (v_hat.sqrt() + self.eps)
                
                # 权重衰减（可选）
                if self.weight_decay > 0:
                    update.add_(param, alpha=self.weight_decay)
                
                # 信任比率
                param_norm = torch.norm(param)
                update_norm = torch.norm(update)
                
                if param_norm > 0 and update_norm > 0:
                    trust_ratio = param_norm / update_norm
                    # 限制信任比率范围
                    trust_ratio = torch.clamp(trust_ratio, 0.1, 10.0)
                else:
                    trust_ratio = 1.0
                
                param.sub_(self.lr * trust_ratio * update)
```

### LAMB 的核心贡献

- **64K batch size 有效**：在 BERT 预训练中，LAMB 使用 64K batch size 达到与 256 batch size 相同的精度，训练时间从 3 天减少到 76 分钟
- **逐层自适应**：不同层的参数范数差异很大，LAMB 自动为每层选择合适的更新步长

---

## 优化器对比表格

| 特性 | SGD+Momentum | AdaGrad | RMSProp | Adam | AdamW | LAMB |
|------|-------------|---------|---------|------|-------|------|
| 提出时间 | 1986/2013 | 2011 | 2012 | 2015 | 2019 | 2020 |
| 自适应 LR | 否 | 是 | 是 | 是 | 是 | 是 |
| 动量 | 是 | 否 | 否 | 是（一阶矩） | 是 | 是 |
| 二阶矩 | 否 | 梯度平方和 | 梯度平方 EMA | 梯度平方 EMA | 梯度平方 EMA | 梯度平方 EMA |
| 权重衰减 | 手动 | 手动 | 手动 | 耦合 | 解耦 | 可选 |
| 大 batch 训练 | 差 | 差 | 差 | 差（LR 需调低） | 一般 | 优秀 |
| 收敛速度 | 慢 | 中等 | 快 | 快 | 快 | 快 |
| 泛化性能 | 好 | 一般 | 一般 | 一般 | 好 | 好 |
| 超参数数 | 2 (lr, μ) | 1 (lr) | 2 (lr, β) | 4 (lr, β₁, β₂, ε) | 5 (+wd) | 5 (+wd) |
| 使用场景 | CNN | 稀疏特征 | RNN | 通用 | LLM 预训练 | 大 batch 预训练 |
| 代表模型 | ResNet | — | — | GPT-2 | BERT, GPT-3 | BERT (大 batch) |

---

## 面试问答

### Q1: AdamW 为什么比 Adam 的泛化性能更好？

**A**: 核心原因在于**权重衰减与自适应学习率的解耦**：

**原始 Adam 的权重衰减问题**：
- Adam 将权重衰减项 $\lambda \theta$ 加入梯度中：$g' = g + \lambda \theta$
- 然后这项的衰减被除以 $\sqrt{\hat{v}_t}$，即自适应学习率缩放
- 结果：频繁更新的参数（小 $v$）权重衰减更强，稀疏更新的参数（大 $v$）权重衰减更弱
- 这破坏了权重衰减的意图——**所有参数应该被同等地正则化**

**AdamW 的解耦方案**：
- 权重衰减独立于梯度更新：$\theta \leftarrow \theta - \eta \lambda \theta$（直接在参数上执行）
- 与 Adam 的自适应更新完全分离
- 所有参数使用**相同的权重衰减率**

**实验证据**：
- Loshchilov & Hutter 在多个任务上验证了 AdamW 优于 Adam + L2 regularization
- GPT-3、LLaMA、BERT 等几乎所有现代 LLM 都使用 AdamW

### Q2: 为什么 LAMB 可以支持大规模 batch 训练？

**A**: LAMB 通过**信任比率（Trust Ratio）**解决了大 batch 训练的核心困难：

**大 batch 训练的挑战**：当 batch size 增加时，梯度噪声降低，理论上可以使用更大的学习率加速训练。但实践中，使用 Adam 或 SGD 增大学习率时，某些层（通常是浅层）的参数更新不稳定，导致训练发散。

**LAMB 的解决方案**：
1. **逐层信任比率**：$r_t = ||\theta|| / ||\text{update}||$ 衡量当前更新相对于参数范数的"冲击大小"。约束 $r_t$ 在 0.1-10 范围内，确保更新不会过大或过小。
2. **层自适应**：浅层（低 $||\theta||$）和深层（高 $||\theta||$）自动获得合适的信任比率。

**效果**：LAMB 在 64K batch size 下训练 BERT，仅 76 分钟达到与 baseline 相同的精度（原需 3 天）。这是因为它允许将学习率从 Adam 的 1e-4 提升到 0.004（40 倍），同时保持训练稳定。

### Q3: 如何选择优化器？

**A**: 根据任务类型和计算资源选择：

| 场景 | 推荐优化器 | 理由 |
|------|-----------|------|
| 小规模实验（<1B 参数） | **Adam** | 收敛快，超参数鲁棒，开箱即用 |
| LLM 预训练（>1B 参数） | **AdamW** | 更好的泛化，解耦的权重衰减 |
| 超大 batch 训练 | **LAMB** | 支持 64K+ batch size 稳定训练 |
| CNN 图像分类 | **SGD + Momentum** | 简单、泛化好、省显存 |
| 稀疏特征场景（NLP 词嵌入） | **AdaGrad** | 自动调节低频特征的 LR |
| 强化学习 | **Adam** | 处理非平稳目标函数 |

**通用建议**：不确定时，使用 AdamW（lr=1e-5~3e-4, wd=0.01）作为起点。对于 LLM 训练，AdamW + cosine LR schedule + weight decay 0.1 是 LLaMA 验证过的可靠配置。

---

## 参考文献

1. Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
2. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 (AdamW)
3. You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", ICLR 2020 (LAMB)
4. Duchi et al., "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", JMLR 2011 (AdaGrad)
5. Tieleman & Hinton, "Lecture 6.5-rmsprop", COURSERA 2012 (RMSProp)
6. Sutskever et al., "On the importance of initialization and momentum in deep learning", ICML 2013 (Nesterov)
7. Zhang et al., "Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model", NeurIPS 2019
