---
# https://vitepress.dev/reference/default-theme-home-page
layout: home


hero:
  name: "大模型知识库"
  text: "汇聚大模型相关的书籍、文章和资源"
  tagline: "探索前沿，理解大模型的世界"
  actions:
    - theme: brand
      text: "书籍与资料"
      link: /books-resources
    - theme: alt
      text: "文章与研究"
      link: /articles-research
---

# 书籍列表

<div class="book-list">
  <div class="book-card">
    <a href="/bllm/">
      <img src="./bllm/images/cover.png" alt="技术书 A" class="book-cover" />
    </a>
    <p>这是一本关于技术主题的书，包含详细的技术文章。</p>
  </div>
</div>

<style>
    .book-list {
    margin-top: 20px;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  
  .book-card {
    width: 200px;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease-in-out;
  }

  .book-card:hover {
    transform: translateY(-5px);
  }

  .book-cover {
    width: 100%;
    height: auto;
  }

  .book-card h3 {
    font-size: 1.2rem;
    margin: 10px 0;
  }

  .book-card p {
    padding: 0 10px 10px;
    font-size: 0.9rem;
    color: #555;
  }

  .book-card a {
    text-decoration: none;
    color: inherit;
  }

</style>
---