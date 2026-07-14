---
title: "归一化技术"
excerpt: "Layer Norm、Batch Norm、RMS Norm 的原理、区别与应用场景。"
date: 2024-06-15
category: "deep-learning"
tags: ["normalization", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

归一化将中间层激活值调整到合适尺度，防止梯度消失/爆炸，加速训练收敛。

## Batch Normalization

沿 batch 维度归一化：$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

适用于 CV 任务，但对 batch size 敏感，在 NLP 和 RNN 中表现不佳。

## Layer Normalization

沿特征维度归一化，不受 batch size 影响，是 Transformer 的标准选择。

$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$

## RMS Norm

简化版 Layer Norm，只做均方根归一化，不含均值中心化：

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma $$

LLaMA 等模型广泛使用，省去均值计算开销。

## Pre-Norm vs Post-Norm

- **Post-Norm**: 子层后归一化（原始 Transformer），训练不稳定
- **Pre-Norm**: 子层前归一化，训练更稳定，现代模型默认选择
