---
title: "激活函数"
excerpt: "ReLU、GELU、SwiGLU 的定义、特性与在 LLM 中的选择。"
date: 2024-06-15
category: "deep-learning"
tags: ["activation", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

激活函数为神经网络引入非线性，是模型表达能力的核心。

## ReLU

$$ \text{ReLU}(x) = \max(0, x) $$

计算简单、梯度不饱和，但存在 dying ReLU 问题。

## GELU

$$ \text{GELU}(x) = x \cdot \Phi(x) $$

BERT、GPT 等模型广泛使用，在负数区域保留小梯度。

## SwiGLU

SwiGLU 是门控线性单元的变体，LLaMA、PaLM 等模型采用：

$$ \text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV) $$

参数量约为标准 FFN 的 1.5 倍，但效果更好。

## 对比

| 函数 | 计算量 | 梯度特性 | 常用模型 |
|------|-------|---------|---------|
| ReLU | 低 | 负半轴零梯度 | 早期 CNN |
| GELU | 中 | 平滑非饱和 | BERT, GPT |
| SwiGLU | 高 | 门控调节 | LLaMA, PaLM |
