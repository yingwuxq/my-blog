import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import tailwindcss from "@tailwindcss/vite";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const site =
  process.env.SITE_URL || process.env.PUBLIC_SITE_URL || "https://yingwu.com";

const base = process.env.ASTRO_BASE || "";

export default defineConfig({
  site,
  base,
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    syntaxHighlight: "shiki",
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
      wrap: false,
    },
  },
  integrations: [mdx()],
  vite: {
    plugins: [tailwindcss()],
  },
});
