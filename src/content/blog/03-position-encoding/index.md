---
title: "位置编码"
excerpt: "Sinusoidal、RoPE、ALiBi 等位置编码的原理与对比。"
date: 2024-06-15
category: "deep-learning"
tags: ["position-encoding", "transformer"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

Self-Attention 是置换等变的，若不注入位置信息，模型无法区分 token 顺序。位置编码就是为了解决这一问题。

## Sinusoidal 位置编码

使用固定频率的正弦/余弦函数：

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) $

$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}) $$

优点是外推能力强，无需学习参数。

## 可学习位置编码

将位置编码作为可训练参数，如 BERT 和 GPT 使用的做法。

## Rotary Position Embedding (RoPE)

通过旋转矩阵对 $Q$ 和 $K$ 进行变换，使内积自然包含相对位置信息。LLaMA、Mistral 等模型采用。

$$ \text{RoPE}(x_m, m) = R_{\Theta,m} x_m $$

## ALiBi

直接给注意力分数加上线性偏置，越远的位置偏置越大，无需额外参数。

## 对比

| 方法 | 外推能力 | 参数 | 常用模型 |
|------|---------|------|---------|
| Sinusoidal | 强 | 无 | Transformer |
| 可学习 | 弱 | $n \times d$ | BERT, GPT |
| RoPE | 强 | 无 | LLaMA, Mistral |
| ALiBi | 强 | 无 | BLOOM |
