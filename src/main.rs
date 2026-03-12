use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sled::Db;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "imgvec",
    version,
    about = "将图片 embedding 到本地嵌入式向量库，并支持自然语言搜索"
)]
struct Cli {
    /// 数据库目录，默认 .imgvec_db
    #[arg(long, default_value = ".imgvec_db")]
    db: PathBuf,

    /// Gemini API base URL
    #[arg(long, default_value = "https://generativelanguage.googleapis.com")]
    api_base: String,

    /// Gemini embedding 模型名称
    #[arg(long, default_value = "gemini-embedding-2-preview")]
    model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 对目录中的图片建索引
    Index(IndexArgs),
    /// 使用自然语言搜索图片
    Search(SearchArgs),
    /// 查看数据库状态
    Stats,
}

#[derive(Args, Debug)]
struct IndexArgs {
    /// 图片目录
    dir: PathBuf,

    /// 递归扫描子目录
    #[arg(short, long, default_value_t = true)]
    recursive: bool,

    /// 强制重建 embedding（忽略 hash 缓存）
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(Args, Debug)]
struct SearchArgs {
    /// 自然语言查询
    query: String,

    /// 返回结果数量
    #[arg(short, long, default_value_t = 10)]
    top_k: usize,

    /// 最小相似度阈值（-1.0 到 1.0）
    #[arg(long, default_value_t = -1.0)]
    min_score: f32,

    /// JSON 输出
    #[arg(long, default_value_t = false)]
    json: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageRecord {
    path: String,
    sha256: String,
    modified_unix_secs: u64,
    mime_type: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedRequest {
    model: String,
    content: Content,
    #[serde(rename = "taskType")]
    task_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    embedding: Option<VectorValues>,
    embeddings: Option<Vec<VectorValues>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorValues {
    values: Vec<f32>,
}

struct GeminiClient {
    http: Client,
    api_base: String,
    model: String,
    api_key: String,
}

impl GeminiClient {
    fn new(api_base: String, model: String, api_key: String) -> Self {
        Self {
            http: Client::new(),
            api_base,
            model,
            api_key,
        }
    }

    fn embed_image(&self, path: &Path) -> Result<Vec<f32>> {
        let bytes = fs::read(path).with_context(|| format!("读取图片失败: {}", path.display()))?;
        let mime_type = detect_mime(path);
        let req = EmbedRequest {
            model: format!("models/{}", self.model),
            content: Content {
                parts: vec![Part::InlineData {
                    inline_data: InlineData {
                        mime_type,
                        data: BASE64.encode(bytes),
                    },
                }],
            },
            task_type: "RETRIEVAL_DOCUMENT".to_string(),
        };
        self.embed(req)
    }

    fn embed_text_query(&self, query: &str) -> Result<Vec<f32>> {
        let req = EmbedRequest {
            model: format!("models/{}", self.model),
            content: Content {
                parts: vec![Part::Text {
                    text: query.to_string(),
                }],
            },
            task_type: "RETRIEVAL_QUERY".to_string(),
        };
        self.embed(req)
    }

    fn embed(&self, req: EmbedRequest) -> Result<Vec<f32>> {
        let endpoint = format!(
            "{}/v1beta/models/{}:embedContent?key={}",
            self.api_base, self.model, self.api_key
        );

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let resp = self
            .http
            .post(&endpoint)
            .headers(headers)
            .json(&req)
            .send()
            .context("调用 Gemini embedding API 失败")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .unwrap_or_else(|_| "<无法读取错误响应>".to_string());
            return Err(anyhow!("Gemini API 返回错误: {} - {}", status, body));
        }

        let parsed: EmbedResponse = resp.json().context("解析 embedding 响应失败")?;
        if let Some(embedding) = parsed.embedding {
            return Ok(embedding.values);
        }

        if let Some(mut embeddings) = parsed.embeddings {
            if let Some(first) = embeddings.pop() {
                return Ok(first.values);
            }
        }

        Err(anyhow!("embedding 响应中未包含向量"))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db =
        sled::open(&cli.db).with_context(|| format!("打开数据库失败: {}", cli.db.display()))?;

    match cli.command {
        Commands::Index(args) => {
            let api_key = std::env::var("GEMINI_API_KEY")
                .context("缺少 GEMINI_API_KEY 环境变量，无法调用 embedding API")?;
            let client = GeminiClient::new(cli.api_base, cli.model, api_key);
            run_index(&db, &client, args)
        }
        Commands::Search(args) => {
            let api_key = std::env::var("GEMINI_API_KEY")
                .context("缺少 GEMINI_API_KEY 环境变量，无法调用 embedding API")?;
            let client = GeminiClient::new(cli.api_base, cli.model, api_key);
            run_search(&db, &client, args)
        }
        Commands::Stats => run_stats(&db),
    }
}

fn run_index(db: &Db, client: &GeminiClient, args: IndexArgs) -> Result<()> {
    let image_paths = collect_image_paths(&args.dir, args.recursive)?;
    if image_paths.is_empty() {
        println!("未找到图片文件: {}", args.dir.display());
        return Ok(());
    }

    let tree = db.open_tree("images")?;
    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )?
        .progress_chars("##-"),
    );

    let mut indexed = 0usize;
    let mut skipped = 0usize;

    for path in image_paths {
        pb.set_message(path.display().to_string());
        let record = build_or_skip_record(&tree, client, &path, args.force)?;
        if let Some(record) = record {
            let key = normalize_path(&path)?;
            tree.insert(key.as_bytes(), serde_json::to_vec(&record)?)?;
            indexed += 1;
        } else {
            skipped += 1;
        }
        pb.inc(1);
    }

    tree.flush()?;
    pb.finish_and_clear();
    println!("完成: 新增/更新 {}，跳过 {}", indexed, skipped);
    Ok(())
}

fn run_search(db: &Db, client: &GeminiClient, args: SearchArgs) -> Result<()> {
    if args.top_k == 0 {
        return Err(anyhow!("top_k 必须大于 0"));
    }

    let query_vec = client.embed_text_query(&args.query)?;
    let tree = db.open_tree("images")?;

    let mut scored: Vec<(ImageRecord, f32)> = Vec::new();
    for item in tree.iter() {
        let (_, value) = item?;
        let record: ImageRecord = serde_json::from_slice(&value)?;
        if record.embedding.is_empty() {
            continue;
        }
        let score = cosine_similarity(&query_vec, &record.embedding);
        if score >= args.min_score {
            scored.push((record, score));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    if args.json {
        let out: Vec<serde_json::Value> = scored
            .into_iter()
            .take(args.top_k)
            .map(|(record, score)| {
                serde_json::json!({
                    "path": record.path,
                    "score": score,
                    "mime_type": record.mime_type
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        for (idx, (record, score)) in scored.into_iter().take(args.top_k).enumerate() {
            println!(
                "{:>2}. {:.4}  {}  ({})",
                idx + 1,
                score,
                record.path,
                record.mime_type
            );
        }
    }

    Ok(())
}

fn run_stats(db: &Db) -> Result<()> {
    let tree = db.open_tree("images")?;
    let count = tree.len();
    println!("向量库记录数: {}", count);
    Ok(())
}

fn build_or_skip_record(
    tree: &sled::Tree,
    client: &GeminiClient,
    path: &Path,
    force: bool,
) -> Result<Option<ImageRecord>> {
    let key = normalize_path(path)?;
    let metadata = fs::metadata(path)?;
    let modified_unix_secs = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let current_hash = sha256_hex(path)?;

    if !force {
        if let Some(existing) = tree.get(key.as_bytes())? {
            let existing: ImageRecord = serde_json::from_slice(&existing)?;
            if existing.sha256 == current_hash {
                return Ok(None);
            }
        }
    }

    let embedding = client.embed_image(path)?;
    let record = ImageRecord {
        path: key,
        sha256: current_hash,
        modified_unix_secs,
        mime_type: detect_mime(path),
        embedding,
    };

    Ok(Some(record))
}

fn collect_image_paths(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Err(anyhow!("目录不存在: {}", dir.display()));
    }

    let mut out = Vec::new();
    let max_depth = if recursive { usize::MAX } else { 1 };

    for entry in WalkDir::new(dir)
        .follow_links(false)
        .max_depth(max_depth)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() && is_supported_image(path) {
            out.push(path.to_path_buf());
        }
    }

    out.sort();
    Ok(out)
}

fn is_supported_image(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    matches!(
        ext.as_str(),
        "jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp" | "tiff" | "heic" | "heif"
    )
}

fn detect_mime(path: &Path) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .unwrap_or("application/octet-stream")
        .to_string()
}

fn normalize_path(path: &Path) -> Result<String> {
    let abs = fs::canonicalize(path)?;
    Ok(abs.to_string_lossy().into_owned())
}

fn sha256_hex(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return -1.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return -1.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_basic() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn image_extension_filter() {
        assert!(is_supported_image(Path::new("a.JPG")));
        assert!(is_supported_image(Path::new("a.png")));
        assert!(!is_supported_image(Path::new("a.txt")));
    }
}
