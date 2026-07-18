const siteUrl = (
  process.env.SITE_URL ||
  import.meta.env?.PUBLIC_SITE_URL ||
  "https://yingwu.com"
).replace(/\/$/, "");

export const SITE = {
  name: "Yingwu'Log",
  description: "记录深度学习、机器学习与工程实践的个人博客。",
  url: siteUrl,
  locale: "zh-CN",
  language: "zh",
  repositoryUrl: "https://github.com/yingwuxq/my-blog",
};

export const NAVIGATION = [
  { to: "/", label: "首页" },
  { to: "/blog", label: "文章" },
  { to: "/tags", label: "标签" },
  { to: "/about", label: "关于" },
];

export const CONTACT = {
  email: "mayingwu0@gmail.com",
  socialHandle: "Ma Yingwu",
  socialUrl: "https://github.com/yingwuxq",
};

export const FORMS = {
  contact: {
    action: "",
    method: "post",
    enctype: "application/x-www-form-urlencoded",
  },
  newsletter: {
    action: "",
    method: "post",
    enctype: "application/x-www-form-urlencoded",
  },
};

export const SOCIAL_LINKS = [{ href: "/rss.xml", label: "RSS feed", icon: "rss" }];

export const authors = [
  {
    slug: "yingwu",
    name: "Ma Yingwu",
    bio: "一名在 AI 世界里认真摸鱼、在 Agent 赛道上持续进修的人工智能专业大学生。",
    longBio: "记录深度学习、机器学习和工程实践的笔记与思考。",
    avatar: "/avatars/yingwu.svg",
  },
];

export const categories = [
  { slug: "deep-learning", name: "深度学习" },
  { slug: "machine-learning", name: "机器学习" },
  { slug: "engineering", name: "工程实践" },
  { slug: "math", name: "数学基础" },
];

export const tags = [
  { slug: "transformer", name: "Transformer" },
  { slug: "attention", name: "Attention" },
  { slug: "deep-learning", name: "Deep Learning" },
  { slug: "position-encoding", name: "Position Encoding" },
  { slug: "normalization", name: "Normalization" },
  { slug: "activation", name: "Activation" },
  { slug: "loss-function", name: "Loss Function" },
  { slug: "optimizer", name: "Optimizer" },
  { slug: "regularization", name: "Regularization" },
  { slug: "initialization", name: "Initialization" },
  { slug: "linear-algebra", name: "Linear Algebra" },
  { slug: "mathematical-foundations", name: "Mathematical Foundations" },
];
