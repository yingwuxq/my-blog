# 激活函数

## 概述

激活函数（Activation Function）为神经网络引入**非线性**，是模型表达能力的核心来源。没有激活函数，多层线性变换的叠加仍然是一个线性变换。本章深入探讨 Transformer 和现代 LLM 中使用的关键激活函数。

---

## 1. ReLU 系列

### ReLU（Rectified Linear Unit）

$$\text{ReLU}(x) = \max(0, x)$$

**优点**：
- 计算极简单（一次比较操作）
- 梯度在 $x > 0$ 时为常数 1，不存在饱和区梯度消失问题
- 稀疏激活（$x < 0$ 区域输出为 0）

**缺点**：
- **Dying ReLU**：当参数更新后某个神经元对所有输入都输出为负，该神经元梯度永远为 0，永远无法恢复
- 非零中心（输出均值为正），可能影响梯度效率

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ReLU 实现
relu = nn.ReLU()
x = torch.linspace(-5, 5, 100)
y = relu(x)

# 可视化
# plt.plot(x.numpy(), y.numpy(), label='ReLU')
# plt.grid(True); plt.legend(); plt.show()

# Dying ReLU 问题示例
class DyingReLUDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))

# 如果所有输入都导致负值，神经元将永久死亡
model = DyingReLUDemo()
with torch.no_grad():
    # 制造一个"死亡"的权重矩阵
    model.fc.weight.fill_(-1.0)
    model.fc.bias.fill_(-1.0)

x_test = torch.randn(32, 10)
out = model(x_test)
print(f"Dead neurons (always zero): {(out == 0).all(dim=0).sum().item()} / 100")
# 输出: Dead neurons (always zero): 100 / 100
```

### LeakyReLU

$$\text{LeakyReLU}(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}$$

其中 $\alpha$ 通常取 0.01。在 $x \leq 0$ 时仍然有斜率 $\alpha$，避免了 Dying ReLU 问题。

```python
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x):
        return torch.where(x > 0, x, self.negative_slope * x)

# 验证
leaky_relu = LeakyReLU(0.01)
x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 5.0])
print(f"LeakyReLU({x}): {leaky_relu(x)}")
# 输出: tensor([-0.0200, -0.0050,  0.0000,  1.0000,  5.0000])
```

### PReLU（Parametric ReLU）

$$\text{PReLU}(x) = \begin{cases} x, & x > 0 \\ a x, & x \leq 0 \end{cases}$$

$\alpha$ 是**可学习**的参数，每个通道可以有不同的 $\alpha$。

```python
# PReLU 可学习的负斜率
prelu = nn.PReLU(num_parameters=1)  # 所有通道共享一个 alpha
prelu_per_channel = nn.PReLU(num_parameters=64)  # 每个通道独立的 alpha

# 查看可学习参数
print(f"PReLU alpha (before): {prelu.weight.item():.4f}")

# 训练过程中 alpha 会被梯度更新
x = torch.randn(32, 64)
out = prelu_per_channel(x)
loss = out.sum()
loss.backward()
print(f"PReLU alpha grad: {prelu_per_channel.weight.grad}")
```

### ELU（Exponential Linear Unit）

$$\text{ELU}(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \leq 0 \end{cases}$$

ELU 在负区域的输出趋近于 $-\alpha$（而不是 0），使得输出均值更接近于 0，有利于梯度流动。但计算量比 ReLU 大。

---

## 2. GELU（Gaussian Error Linear Unit）

### 定义

GELU（Hendrycks & Gimpel, 2016）是 BERT 和 GPT 系列使用的激活函数，它结合了 ReLU 的稀疏性和 Dropout 的正则化思想：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x)$$

其中 $\Phi(x)$ 是标准正态分布 $N(0,1)$ 的累积分布函数（CDF）。

**直觉**：GELU 以概率 $\Phi(x)$"保留"输入 $x$，以概率 $1-\Phi(x)$"关闭"它（乘以 0）。这比 ReLU 的硬截断更加平滑。

### 精确公式

$$\text{GELU}(x) = \frac{x}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\text{erf}$ 是误差函数。

### 近似实现

由于 $\text{erf}$ 计算较慢，实际使用中常用 Tanh 近似：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right]\right)$$

或者使用 Sigmoid 近似（GPT-2 使用的版本）：

$$\text{GELU}(x) \approx x \cdot \sigma(1.702 x)$$

```python
def gelu_exact(x):
    """精确 GELU 实现"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_approx_tanh(x):
    """Tanh 近似 GELU（计算更快）"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def gelu_approx_sigmoid(x):
    """Sigmoid 近似 GELU"""
    return x * torch.sigmoid(1.702 * x)

# 对比误差
x = torch.linspace(-5, 5, 1000)
exact = gelu_exact(x)
tanh_approx = gelu_approx_tanh(x)
sig_approx = gelu_approx_sigmoid(x)

print(f"Tanh approx max error: {(exact - tanh_approx).abs().max():.6f}")
print(f"Sigmoid approx max error: {(exact - sig_approx).abs().max():.6f}")
# 输出示例:
# Tanh approx max error: 0.000032
# Sigmoid approx max error: 0.001032
```

### GELU 的梯度

$$\frac{\partial}{\partial x} \text{GELU}(x) = \Phi(x) + x \cdot \phi(x) = \Phi(x) + x \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

其中 $\phi(x)$ 是标准正态分布的 PDF。这个梯度公式比 ReLU 的阶跃型梯度更平滑，有助于更稳定的训练。

---

## 3. SwiGLU / SiLU

### SiLU（Sigmoid Linear Unit）

SiLU（Elfwing et al., 2018），也称为 Swish（Ramachandran et al., 2017）：

$$\text{SiLU}(x) = \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**特点**：
- 平滑、非单调（在 $x \approx -1.28$ 处有最小值）
- 无上界、有下界
- 自门控（self-gating）：通过 $\sigma(x)$ 自动调节信息流动

```python
def silu(x):
    """SiLU / Swish 激活函数"""
    return x * torch.sigmoid(x)

# 梯度计算
x = torch.tensor([-3.0, -1.28, 0.0, 2.0, 5.0], requires_grad=True)
y = silu(x).sum()
y.backward()
print(f"x: {x.detach()}")
print(f"SiLU(x): {silu(x.detach())}")
print(f"grad: {x.grad}")
```

### SwiGLU（Swish-Gated Linear Unit）

SwiGLU（Shazeer, 2020）结合了 GLU（Gated Linear Unit）和 SiLU：

$$\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_2) W_3$$

其中 $\odot$ 是逐元素乘法（Hadamard 积），$W_1, W_2, W_3$ 是权重矩阵。

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数（LLaMA 中使用的版本）
    
    与标准 FFN 不同的是，SwiGLU 使用三个权重矩阵。
    通常保持参数量不变的方法是：将 d_ff 从 4d 调整为 ~8/3 d
    """
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        # SwiGLU 需要三个投影（标准 FFN 只需要两个）
        # 通常 d_ff 取 (2/3) * 4 * d_model ≈ 2.67 * d_model
        d_ff = d_ff or int(8 * d_model / 3)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        # SiLU(xW1) * (xW2) 作为门控机制
        gate = F.silu(self.W1(x))  # 门控信号
        value = self.W2(x)         # 值信号
        return self.W3(gate * value)  # 门控乘法 + 输出投影

# 参数量比较
d_model = 512

# 标准 FFN: 2 * d_model * d_ff = 2 * 512 * 2048 = 2,097,152
std_ffn = nn.Sequential(
    nn.Linear(d_model, 2048),
    nn.ReLU(),
    nn.Linear(2048, d_model)
)
std_params = sum(p.numel() for p in std_ffn.parameters())

# SwiGLU FFN: d_model * d_ff * 3 = 512 * 1365 * 3 = 2,096,640
swiglu_ffn = SwiGLU(d_model, d_ff=1365)
swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"Standard FFN params: {std_params:,}")
print(f"SwiGLU FFN params:   {swiglu_params:,}")
print(f"Difference:          {abs(std_params - swiglu_params):,}")
# 输出: 两种方案的参数量几乎相同
```

### 为什么 LLaMA 选择 SwiGLU？

SwiGLU 在 PaLM 和 LLaMA 中被证明在相同参数量下，**相比 ReLU FFN 和 GELU FFN 有更好的性能**：

| 特性 | GELU FFN | SwiGLU FFN |
|------|----------|------------|
| 权重矩阵数 | 2 个 | 3 个 |
| 参数量调整 | 标准 $d_{ff}=4d$ | $d_{ff} \approx \frac{8}{3}d$ |
| 与 ReLU FFN 等参数量 | $2 \times 4d^2 = 8d^2$ | $3 \times \frac{8}{3}d^2 = 8d^2$ |
| 语言建模困惑度 | 基准 | 更好 |
| 门控机制 | 无 | 有（SiLU gate） |

---

## 4. 对比表格

| 激活函数 | 公式 | 梯度特性 | 输出范围 | 计算开销 | 使用模型 |
|----------|------|---------|---------|---------|---------|
| ReLU | $\max(0,x)$ | $x>0: 1$, $x\leq0: 0$ | $[0, \infty)$ | 极低 | 原始 Transformer |
| LeakyReLU | $\max(\alpha x, x)$ | $x>0: 1$, $x\leq0: \alpha$ | $(-\infty, \infty)$ | 低 | GAN |
| GELU | $x \cdot \Phi(x)$ | $\Phi(x) + x\phi(x)$ 平滑 | $(-\infty, \infty)$ | 中 | BERT, GPT-2 |
| SiLU / Swish | $x \cdot \sigma(x)$ | 平滑非单调 | $(-\infty, \infty)$ | 低 | 部分 ViT |
| SwiGLU | $\text{SiLU}(xW_1) \odot xW_2$ | 带有门控乘法 | $(-\infty, \infty)$ | 中高 | LLaMA, PaLM |
| ELU | $x>0: x; x\leq0: \alpha(e^x-1)$ | $x>0: 1$, $x\leq0: \alpha e^x$ | $(-\alpha, \infty)$ | 中 | 少量研究 |

### 训练稳定性对比

| 特性 | ReLU | GELU | SwiGLU |
|------|------|------|--------|
| Dying ReLU 问题 | 严重 | 无 | 无 |
| 梯度平滑度 | 阶跃型（不连续） | 平滑 | 平滑 |
| 均值偏移 | 正偏移 | 近零 | 近零 |
| 深层网络训练 | 需 careful init | 较稳定 | 最稳定 |

---

## 面试问答

### Q1: GELU 相比于 ReLU 有什么优势？

**A**: GELU 相比 ReLU 有三个核心优势：

1. **平滑梯度**：ReLU 在 $x=0$ 处不可导（梯度从 0 跳变到 1），而 GELU 处处可导且梯度连续。平滑的梯度使得训练更稳定，尤其在没有梯度裁剪或 warmup 的情况下。

2. **概率门控**：GELU 乘以 $\Phi(x)$（概率值），相当于一个"软"开关——当 $x=-0.5$ 时不是完全关闭（0），而是保留约 30%。这种概率解释天然具备 Dropout 风格的正则化效果。

3. **经验验证**：BERT、GPT-2、ViT 等模型一致地报告 GELU 优于 ReLU。在语言建模任务上，GELU 通常比 ReLU 降低 0.5-1.0 的 perplexity。

### Q2: 为什么 LLaMA 使用 SwiGLU 而不是 GELU？

**A**: SwiGLU 相对于 GELU 的核心差异在于**门控机制**：

1. **门控线性单元**：SwiGLU 的 $\text{SiLU}(xW_1) \odot (xW_2)$ 引入了一个显式的门控机制——$\text{SiLU}(xW_1)$ 控制第二个投影 $(xW_2)$ 中哪些信息可以通过。这种门控被证明比简单的逐元素激活函数（如 GELU）有更强的表达能力。

2. **等价参数量下的更好性能**：虽然 SwiGLU 有三个权重矩阵（标准 FFN 只有两个），但通过将 $d_{ff}$ 从 $4d$ 缩小到 $\frac{8}{3}d$，总参数量保持不变。在此条件下，SwiGLU 的语言建模 perplexity 优于 GELU 和 ReLU。

3. **经验结果**：PaLM 论文（Chowdhery et al., 2022）报告，在等参数量设置下，SwiGLU 在所有评估任务上均优于 ReLU 和 GELU。LLaMA 论文也确认了这一点。

**注意**：SwiGLU 的训练速度略慢于 GELU FFN（因为多了第二个投影矩阵乘法 $xW_2$），但性能收益被认为值得这个代价。

### Q3: SiLU/Swish 为什么称为"自门控"激活函数？

**A**: SiLU/Swish 被称为"自门控"是因为它的门控信号来自于**输入本身**：

$$\text{Swish}(x) = x \cdot \sigma(x)$$

这里的 $\sigma(x)$（sigmoid）可以被视为一个**依赖于 $x$ 的门控值**：

- 当 $x$ 很大时（$x \gg 0$），$\sigma(x) \approx 1$，SWish(x) $\approx x$（信息完全通过）
- 当 $x$ 很小时（$x \ll 0$），$\sigma(x) \approx 0$，Swish(x) $\approx 0$（信息被阻挡）
- 当 $x=0$ 时，$\sigma(0)=0.5$，Swish(0) $= 0$（半开状态）

对比 ReLU：ReLU 的门控值是硬阈值 $\mathbb{I}(x > 0)$，要么 0 要么 1。而 Swish 的门控是平滑的、连续的，因此梯度更加平滑。

SwiGLU 则是"自门控"的扩展——门控信号和值信号来自不同的线性投影，提供了更强的表达能力。

---

## 参考文献

1. Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines", ICML 2010
2. Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv 2016
3. Ramachandran et al., "Searching for Activation Functions", arXiv 2017 (Swish)
4. Elfwing et al., "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning", Neural Networks 2018 (SiLU)
5. Shazeer, "GLU Variants Improve Transformer", arXiv 2020 (SwiGLU)
6. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", 2022
7. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023
8. Clevert et al., "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", ICLR 2016
9. He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet", ICCV 2015 (PReLU)
