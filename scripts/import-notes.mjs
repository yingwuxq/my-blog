import fs from "fs"
import path from "path"

const notesDir = path.join(import.meta.dirname, "..", "note")
const postsDir = path.join(import.meta.dirname, "..", "content", "posts")

const meta = {
  "01-transformer-architecture": {
    title: "Transformer 架构完整详解",
    desc: "从 Encoder-Decoder 结构到 Multi-Head Attention，全面拆解 Transformer 的核心设计。",
    tags: ["transformer", "deep-learning"],
    category: "深度学习",
  },
  "02-attention-mechanisms": {
    title: "注意力机制全系列",
    desc: "从 Scaled Dot-Product Attention 到 FlashAttention，梳理注意力机制的完整进化路线。",
    tags: ["attention", "deep-learning"],
    category: "深度学习",
  },
  "03-position-encoding": {
    title: "位置编码",
    desc: "Sinusoidal、可学习、RoPE、ALiBi — 各种位置编码的原理与对比。",
    tags: ["position-encoding", "transformer"],
    category: "深度学习",
  },
  "04-normalization": {
    title: "归一化技术",
    desc: "Layer Norm、Batch Norm、RMS Norm 等归一化方法的原理、区别与应用场景。",
    tags: ["normalization", "deep-learning"],
    category: "深度学习",
  },
  "05-activation-functions": {
    title: "激活函数",
    desc: "ReLU、GELU、SwiGLU 等激活函数的数学定义、特性与在 LLM 中的选择。",
    tags: ["activation", "deep-learning"],
    category: "深度学习",
  },
  "06-loss-functions": {
    title: "损失函数",
    desc: "交叉熵、对比损失、CTC 损失等，覆盖分类、生成、序列任务的损失设计。",
    tags: ["loss-function", "deep-learning"],
    category: "深度学习",
  },
  "07-optimizers": {
    title: "优化器",
    desc: "Adam、AdamW、SGD 及其变体的原理、公式与训练稳定性分析。",
    tags: ["optimizer", "deep-learning"],
    category: "深度学习",
  },
  "08-regularization": {
    title: "正则化",
    desc: "Dropout、Weight Decay、Label Smoothing 等正则化策略的原理与实践。",
    tags: ["regularization", "deep-learning"],
    category: "深度学习",
  },
  "09-initialization": {
    title: "初始化方法",
    desc: "Xavier、Kaiming 等初始化方法的数学原理与对训练收敛的影响。",
    tags: ["initialization", "deep-learning"],
    category: "深度学习",
  },
}

const files = fs.readdirSync(notesDir).filter((f) => f.endsWith(".md"))

for (const file of files) {
  const baseName = file.replace(/\.md$/, "")
  const content = fs.readFileSync(path.join(notesDir, file), "utf8")
  const m = meta[baseName]

  const frontmatter = [
    "---",
    `title: "${m.title}"`,
    `date: "2024-06-15"`,
    `description: "${m.desc}"`,
    `tags: [${m.tags.map((t) => `"${t}"`).join(", ")}]`,
    `category: "${m.category}"`,
    "---",
    "",
  ].join("\n")

  const body = content.replace(/^# .+\n/, "").trim()

  fs.writeFileSync(path.join(postsDir, `${baseName}.md`), frontmatter + body + "\n")
  console.log(`✅ ${baseName}.md`)
}

console.log("\nAll notes imported!")
