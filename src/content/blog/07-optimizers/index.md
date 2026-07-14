---
title: "优化器"
excerpt: "SGD、Adam、AdamW 的原理与训练稳定性对比。"
date: 2024-06-15
category: "deep-learning"
tags: ["optimizer", "deep-learning"]
author: "yingwu"
featured: false
draft: false
thumbnail: ./cover.webp
---

## 概述

优化器根据梯度更新模型参数，是训练的核心引擎。

## SGD + Momentum

$$ v_{t+1} = \beta v_t + \nabla \mathcal{L}(\theta_t) $

$ \theta_{t+1} = \theta_t - \eta v_{t+1} $$

动量累积历史梯度方向，加速收敛并抑制震荡。

## Adam

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t $

$ v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 $$

自适应学习率，结合动量与 RMSProp 的优点，是 NLP 任务的首选。

## AdamW

在 Adam 基础上将 weight decay 与学习率解耦，LLM 训练的标准选择：

$$ \theta_{t+1} = \theta_t - \eta(\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_t) $$

## 对比

| 优化器 | 收敛速度 | 内存 | 适用场景 |
|-------|---------|------|---------|
| SGD+Momentum | 慢 | 低 | CV、简单任务 |
| Adam | 快 | 高 | NLP、通用 |
| AdamW | 快 | 高 | LLM 预训练 |
