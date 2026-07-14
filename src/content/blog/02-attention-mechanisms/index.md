---
title: "注意力机制"
excerpt: "从 Scaled Dot-Product 到 FlashAttention，注意力机制的进化。"
date: 2024-06-15
category: "deep-learning"
tags: ["attention", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

注意力机制的核心思想是允许模型动态关注输入序列中的相关信息。

## Scaled Dot-Product Attention

标准形式计算 Query 与 Key 的相似度，以此加权 Value。

## 多头注意力 (MHA)

将 $Q, K, V$ 投影到 $h$ 个子空间，分别计算注意力后拼接：

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

## 变体

- **Multi-Query Attention (MQA)**: 所有头共享 $K, V$，减少 KV 缓存
- **Grouped-Query Attention (GQA)**: MHA 与 MQA 的折中，分组共享
- **FlashAttention**: 通过分块计算和 IO 感知优化加速注意力，无需存储完整注意力矩阵

## 对比

| 变体 | 速度 | 内存 | 质量 |
|------|------|------|------|
| MHA | 基准 | 基准 | 基准 |
| MQA | 快 | 低 | 略降 |
| GQA | 较快 | 较低 | 接近 |
| FlashAttention | 快 | 低 | 无损 |
