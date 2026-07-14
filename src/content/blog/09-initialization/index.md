---
title: "初始化方法"
excerpt: "Xavier、Kaiming 等初始化方法的数学原理与影响。"
date: 2024-06-15
category: "deep-learning"
tags: ["initialization", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

好的初始化方法加速收敛、防止梯度消失/爆炸，是训练深层网络的基础。

## Xavier (Glorot) 初始化

$$ W \sim \mathcal{U}(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}) $$

使各层激活值和梯度的方差在传播过程中保持一致。适用于 tanh / sigmoid 激活。

## Kaiming (He) 初始化

$$ W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{\text{in}}}}) $$

针对 ReLU 及其变体设计，考虑 ReLU 将一半神经元置零的方差衰减效应。

## 初始化对比

| 方法 | 适用激活 | 方差公式 | 特点 |
|------|---------|---------|------|
| Xavier | tanh, sigmoid | $\frac{2}{n_{\text{in}}+n_{\text{out}}}$ | 保持方差恒定 |
| Kaiming | ReLU, GELU | $\frac{2}{n_{\text{in}}}$ | 适配非线性 |

## 实际建议

- Transformer 通常使用 Kaiming 初始化 + 小值初始化（embedding 层）
- 残差分支可初始化为零，便于训练初期梯度传播
- 大模型训练常用 Fan-In 或 Fan-Out 变体
