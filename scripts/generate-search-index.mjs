import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"
import matter from "gray-matter"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const postsDir = path.join(__dirname, "..", "content", "posts")
const outDir = path.join(__dirname, "..", "public")

const files = fs.readdirSync(postsDir).filter((f) => f.endsWith(".md"))
const posts = files.map((f) => {
  const raw = fs.readFileSync(path.join(postsDir, f), "utf8")
  const { data } = matter(raw)
  return {
    slug: f.replace(/\.md$/, ""),
    title: data.title || "",
    description: data.description || "",
    date: data.date || "",
    tags: data.tags || [],
    category: data.category || "",
  }
})

fs.writeFileSync(path.join(outDir, "search-index.json"), JSON.stringify(posts))
console.log(`Generated search index with ${posts.length} posts`)
