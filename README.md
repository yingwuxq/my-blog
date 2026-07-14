# Yingwu'Log

个人技术博客，记录深度学习、机器学习与工程实践。

**网址：** [yingwuxq.github.io/my-blog](https://yingwuxq.github.io/my-blog)

---

## 技术栈

| 层 | 选型 |
|---|---|
| 框架 | Astro 7 |
| 样式 | Tailwind CSS 4 |
| 字体 | Fraunces（标题） + Inter（正文） + JetBrains Mono（代码） |
| 公式 | KaTeX + remark-math |
| 代码高亮 | Shiki（github-light / github-dark） |
| 评论 | Giscus（基于 GitHub Discussions） |
| 图表 | Mermaid |
| 部署 | GitHub Pages（自动构建） |

---

## 写新文章

### 方式一：直接创建（推荐）

在 `src/content/blog/` 下按以下结构创建：

```
src/content/blog/文章英文 slug/
├── index.md         # 文章内容
└── cover.webp       #（可选）文章缩略图
```

`index.md` 的 frontmatter 格式：

```yaml
---
title: "文章标题"
excerpt: "一句话简介"
date: 2024-07-13
category: "deep-learning"
tags: ["transformer", "attention"]
author: "yingwu"
featured: false
draft: false
---
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `title` | 是 | 文章标题 |
| `excerpt` | 是 | 文章简介，显示在卡片和 meta description |
| `date` | 是 | 日期格式 `YYYY-MM-DD` |
| `category` | 是 | 分类 slug，在 `src/config/theme.config.ts` 中定义 |
| `tags` | 否 | 标签数组，同样需要在 theme.config.ts 中定义 |
| `author` | 是 | 固定为 `"yingwu"` |
| `featured` | 否 | `true` 时显示在首页推荐位 |
| `draft` | 否 | `true` 时不会发布 |

### 方式二：从 content/posts/ 转换

在 `content/posts/` 下创建 `.md` 文件，然后运行：

```bash
node scripts/convert-posts.mjs
```

这种方式会自动生成 `/src/content/blog/[slug]/index.md`。

---

## 文章中的图片

图片存放在文章同目录下，在 Markdown 中直接引用文件名：

```
src/content/blog/my-post/
├── index.md
├── architecture.png
└── flow-chart.png
```

```markdown
![架构图](architecture.png)
```

图片会被自动优化（WebP 格式、响应式尺寸）。

---

## 文章缩略图

在文章目录下放一张图片，frontmatter 中引用：

```yaml
thumbnail: ./cover.webp
thumbnailAlt: "缩略图描述"
```

如果未设置缩略图，文章卡片会自动使用首页英雄区的背景图作为默认封面。

建议缩略图尺寸：**1200×630px**（OG 标准比例）。

---

## 分类和标签

在 `src/config/theme.config.ts` 中维护：

```typescript
categories: [
  { slug: "deep-learning", name: "深度学习" },
]

tags: [
  { slug: "transformer", name: "Transformer" },
]
```

新文章使用的分类和标签必须在此文件中定义，否则会 404。

---

## 本地开发

```bash
npm install          # 安装依赖
npm run dev          # 开发模式 → http://localhost:4321
npm run build        # 构建到 dist/
npm run preview      # 预览构建结果
```

---

## Markdown 特性

- **KaTeX 公式**：行内 `$...$`，行间 `$$...$$`
- **Mermaid 图表**：\`\`\`mermaid 代码块
- **代码高亮**：\`\`\`python / \`\`\`js 等自动高亮，支持暗色模式
- **表格**：标准 Markdown 表格
- **任务列表**：`- [x]` 格式

---

## 部署

`git push` 到 `main` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。
