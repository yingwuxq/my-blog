---
title: "正则化"
excerpt: "Dropout、Weight Decay、Label Smoothing 等策略的原理与实践。"
date: 2024-06-15
category: "deep-learning"
tags: ["regularization", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

正则化防止模型过拟合，是训练大模型的关键技术。

## Weight Decay ($L_2$ 正则化)

在损失函数中加入参数范数惩罚：

$$ \mathcal{L}' = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2 $$

等价于每次更新时对参数做衰减：$\theta \leftarrow (1 - \eta\lambda)\theta - \eta\nabla\mathcal{L}$

## Dropout

训练时以概率 $p$ 随机丢弃神经元，相当于训练多个子网络的集成。现代 LLM 通常使用较小的 dropout（$p=0.1$）。

## Label Smoothing

将硬标签 $[0, 1]$ 替换为软标签 $[\frac{\epsilon}{K-1}, 1-\epsilon]$，防止模型过于自信，提升泛化能力。

## Stochastic Depth

随机跳过某些层，加速训练并起到正则化作用。常用于深层 Transformer。

## 对比

| 方法 | 适用位置 | 训练开销 | 推理影响 |
|------|---------|---------|---------|
| Weight Decay | 参数 | 低 | 无 |
| Dropout | 激活 | 中 | 需缩放 |
| Label Smoothing | 标签 | 低 | 无 |
| Stochastic Depth | 层 | 中 | 无 |
