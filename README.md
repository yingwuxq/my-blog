# Yingwu'Log

个人技术博客，记录深度学习、机器学习与工程实践。

**网址：** [yingwuxq.github.io/my-blog](https://yingwuxq.github.io/my-blog)

---

## 技术栈

- **框架：** [Astro](https://astro.build/) 7
- **样式：** [Tailwind CSS](https://tailwindcss.com/) 4
- **字体：** Fraunces + Inter + JetBrains Mono
- **公式：** KaTeX + remark-math
- **代码高亮：** Shiki
- **部署：** GitHub Pages

## 本地开发

```bash
npm install
npm run dev      # 开发模式，localhost:4321
npm run build    # 构建到 dist/
npm run preview  # 预览构建结果
```

## 目录结构

```
src/
├── content/blog/     # 博客文章 (Markdown)
├── config/           # 主题配置
├── components/       # UI 组件
├── layouts/          # 页面布局
├── pages/            # 路由页面
└── lib/              # 工具函数
public/assets/        # 静态资源（图片、视频）
content/posts/        # 原始 .md 文件备份
```

## 写新文章

在 `content/posts/` 下创建 `.md` 文件，然后运行：

```bash
node scripts/convert-posts.mjs  # 转换到 Astro 内容格式
```

或者直接在 `src/content/blog/[slug]/index.md` 创建，frontmatter 格式：

```yaml
---
title: "文章标题"
excerpt: "简介"
date: 2024-07-13
category: "deep-learning"
tags: ["transformer"]
author: "yingwu"
featured: false
---
```
