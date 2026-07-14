---
title: "Transformer 架构详解"
excerpt: "Encoder-Decoder 结构、Multi-Head Attention 与位置编码的核心设计。"
date: 2024-06-15
category: "deep-learning"
tags: ["transformer", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

Transformer 由 Vaswani 等人于 2017 年提出，完全基于注意力机制，摒弃了 RNN 的循环结构，实现了并行计算和长距离依赖捕获。

## Scaled Dot-Product Attention

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

缩放因子 $\sqrt{d_k}$ 防止内积过大导致 softmax 梯度消失。

## Multi-Head Attention

$$ \text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$

多头机制让模型在不同子空间同时学习注意力模式。

## 核心组件

- **Position-wise FFN**: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
- **位置编码**: 正弦/余弦函数编码位置信息
- **残差连接 + Layer Norm**: 每层后接 Add & Norm

## 模型结构

Encoder 和 Decoder 均由 $N$ 层堆叠，每层包含多头自注意力和 FFN。Decoder 额外包含 Masked Self-Attention 和 Cross-Attention。
