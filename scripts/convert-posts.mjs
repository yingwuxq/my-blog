import fs from "fs"
import path from "path"
import matter from "gray-matter"

const postsDir = path.join(process.cwd(), "content", "posts")
const outDir = path.join(process.cwd(), "src", "content", "blog")

const categorySlug = (cat) => {
  const map = {
    "深度学习": "deep-learning",
    "机器学习": "machine-learning",
  }
  return map[cat] || cat
}

const files = fs.readdirSync(postsDir).filter((f) => f.endsWith(".md"))

for (const file of files) {
  const raw = fs.readFileSync(path.join(postsDir, file), "utf8")
  const { data, content } = matter(raw)
  const slug = file.replace(/\.md$/, "")

  const postDir = path.join(outDir, slug)
  fs.mkdirSync(postDir, { recursive: true })

  // Build YAML frontmatter carefully with proper escaping
  const lines = [
    "---",
    `title: ${JSON.stringify(data.title)}`,
    `excerpt: ${JSON.stringify(data.description || "")}`,
    `date: ${data.date}`,
    `category: ${JSON.stringify(categorySlug(data.category || ""))}`,
    `tags: [${(data.tags || []).map((t) => JSON.stringify(t)).join(", ")}]`,
    `author: ${JSON.stringify("yingwu")}`,
    `featured: ${slug.startsWith("00-") ? "true" : "false"}`,
    "draft: false",
    "---",
    "",
    content.trim(),
    "",
  ]

  fs.writeFileSync(path.join(postDir, "index.md"), lines.join("\n"))
  console.log(`Converted: ${file} -> ${slug}/index.md`)
}

console.log(`\nDone! Converted ${files.length} posts.`)
