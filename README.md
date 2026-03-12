# imgvec

一个基于 Rust 的 CLI：

- 扫描目录中的图片
- 调用 `gemini-embedding-2-preview` 生成向量
- 存入本地嵌入式数据库（sled）
- 用自然语言进行语义搜索

## 快速开始

```bash
export GEMINI_API_KEY="<your-key>"
cargo run -- index ./images
cargo run -- search "一张夕阳下有海边的照片"
```

## 命令

### 建索引

```bash
imgvec index <DIR> [--force] [--recursive]
```

- 默认递归扫描
- 通过文件 SHA256 做增量更新（同文件不重复 embedding）
- `--force` 强制重建

### 语义搜索

```bash
imgvec search "你的自然语言查询" --top-k 10 --min-score 0.2
```

### 查看状态

```bash
imgvec stats
```

## 参数

全局参数：

- `--db <PATH>`：数据库目录，默认 `.imgvec_db`
- `--api-base <URL>`：默认 `https://generativelanguage.googleapis.com`
- `--model <NAME>`：默认 `gemini-embedding-2-preview`

## 说明

该工具使用 Gemini 的 `embedContent` 接口，请确保所用 API key 具有对应模型调用权限。
