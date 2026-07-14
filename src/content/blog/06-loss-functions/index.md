---
title: "损失函数"
excerpt: "交叉熵、对比损失、KL 散度的定义与应用。"
date: 2024-06-15
category: "deep-learning"
tags: ["loss-function", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

损失函数度量模型预测与真实目标之间的差距，是模型优化的直接信号。

## 交叉熵损失

$$ \mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) $$

分类任务的标准损失。对于单标签分类简化为：

$$ \mathcal{L}_{\text{CE}} = -\log(\hat{y}_c) $$

## KL 散度

$$ D_{\text{KL}}(P \parallel Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)} $$

度量两个分布之间的差异，常用于蒸馏和 RLHF。

## 对比损失 (InfoNCE)

$$ \mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(z_i^a, z_i^p) / \tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i^a, z_j^p) / \tau)} $$

拉近正样本对、推远负样本对，CLIP、SimCLR 等模型使用。

## 损失对比

| 损失 | 应用场景 | 特性 |
|------|---------|------|
| Cross-Entropy | 分类、语言模型 | 标准选择 |
| InfoNCE | 对比学习 | 需要负样本 |
| KL Divergence | 蒸馏、RLHF | 不对称 |
