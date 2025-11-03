use anyhow::Result;
use fastembed::EmbeddingModel;
use rusty_embeddings::EmbeddingService;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

#[derive(Debug, Clone)]
pub struct GroupParams {
    pub split_threshold: f32,
    pub min_sentences: usize,
    pub max_chars: Option<usize>,
    pub rank_window: usize,
    pub block_depth: usize,
    pub cache_dir: PathBuf,
}

impl GroupParams {
    pub fn hash(&self) -> String {
        let ctx = format!(
            "t{:.2}_ms{}_mc{:?}_rw{}_bd{}",
            self.split_threshold,
            self.min_sentences,
            self.max_chars,
            self.rank_window,
            self.block_depth
        );

        let mut h = blake3::Hasher::new();
        h.update(ctx.as_bytes());
        h.finalize().to_hex().to_string()
    }
}

impl Default for GroupParams {
    fn default() -> Self {
        Self {
            split_threshold: 0.55,
            min_sentences: 2,
            max_chars: None,
            rank_window: 2,
            block_depth: 2,
            cache_dir: PathBuf::from(".cache"),
        }
    }
}

/// groups semantically related sentences into paragraphs
#[derive(Default)]
pub struct ParagraphGrouper {
    params: GroupParams,
}

impl ParagraphGrouper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn split_threshold(mut self, t: f32) -> Self {
        self.params.split_threshold = t;
        self
    }

    pub fn min_sentences(mut self, n: usize) -> Self {
        self.params.min_sentences = n;
        self
    }

    pub fn max_chars(mut self, cap: Option<usize>) -> Self {
        self.params.max_chars = cap;
        self
    }

    pub fn rank_window(mut self, n: usize) -> Self {
        self.params.rank_window = n;
        self
    }

    pub fn block_depth(mut self, n: usize) -> Self {
        self.params.block_depth = n;
        self
    }

    pub fn cache_dir(mut self, dir: PathBuf) -> Self {
        self.params.cache_dir = dir;
        self
    }

    pub fn group(self, sentences: Vec<String>) -> Result<Vec<String>> {
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(hit) = try_load_cached_paragraphs(&self.params, &sentences)? {
            return Ok(hit);
        }

        let embeddings = EmbeddingService::new()
            .model(EmbeddingModel::GTEBaseENV15)
            .cache_dir(self.params.cache_dir.clone())
            .build(sentences.clone())?;

        let paragraphs = group_impl_c99(&sentences, embeddings, &self.params)?;

        let _ = persist_paragraphs_cache(&self.params, &sentences, &paragraphs);

        Ok(paragraphs)
    }
}

fn group_impl_c99(
    sentences: &[String],
    embeddings: Vec<Vec<f32>>,
    params: &GroupParams,
) -> Result<Vec<String>> {
    let n = sentences.len();
    if n == 1 {
        return Ok(vec![sentences[0].clone()]);
    }

    // 1) cosine similarity matrix (since vectors are normalized: cosine == dot)
    let mut s = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        s[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = dot(&embeddings[i], &embeddings[j]);
            s[i][j] = sim;
            s[j][i] = sim;
        }
    }

    // 2) local rank transform: each cell -> percentile rank within a (2w+1)^2 neighborhood
    let r = rank_transform(&s, params.rank_window);

    // 3) boundary scoring along the main diagonal
    //    score(b) = mean(left block) + mean(right block) - 2 * mean(cross blocks)
    //    (higher is a stronger boundary)
    let scores = boundary_scores(&r, params.block_depth);

    // 4) pick boundaries: local maxima above threshold, with guards
    //    (require min_sentences on both sides; respect max_chars if set)
    let mut splits = Vec::<usize>::new(); // positions between b-1 and b
    let min_span = params.min_sentences.max(1);

    // non-maximum suppression window (avoid adjacent peaks)
    let nms = params.block_depth.max(1);

    for b in params.block_depth..(n - params.block_depth) {
        // local maximum?
        let mut is_peak = true;
        let left = b.saturating_sub(nms);
        let right = (b + nms).min(n - 1);
        for k in left..=right {
            if k != b && scores[k] > scores[b] {
                is_peak = false;
                break;
            }
        }

        if !is_peak {
            continue;
        }

        // absolute threshold on boundary score (params.split_threshold in 0..1 works well)
        if scores[b] < params.split_threshold {
            continue;
        }

        // enforce min sentences on both sides
        let left_len = b - splits.last().copied().unwrap_or(0);
        let right_len_min = min_span;
        let remaining_after = n - b;
        if left_len < min_span || remaining_after < right_len_min {
            continue;
        }

        // optional char cap guard: if adding more would exceed cap, force a split
        if let Some(cap) = params.max_chars {
            let cur_start = splits.last().copied().unwrap_or(0);
            let cur_len_chars = joined_len(&sentences[cur_start..b]);
            // boundary is helpful if current paragraph is already near/over cap
            if cur_len_chars < cap && scores[b] < (params.split_threshold + 0.05) {
                // let stronger splits or size pressure decide; skip this mild one
                continue;
            }
        }

        splits.push(b);
    }

    // 5) build paragraphs from splits
    let mut out = Vec::<String>::new();
    let mut start = 0usize;
    for &b in &splits {
        if b > start {
            out.push(sentences[start..b].join(" "));
        }
        start = b;
    }
    if start < n {
        out.push(sentences[start..n].join(" "));
    }

    Ok(out)
}

/// local rank transform:
/// for each cell (i,j), compute the fraction of neighbors in a (2w+1)^2 window
/// whose similarity is <= s[i][j]. this yields values in [0,1], making scores
/// more contrastive and document-scale invariant (c99 trick).
fn rank_transform(s: &[Vec<f32>], w: usize) -> Vec<Vec<f32>> {
    let n = s.len();
    let mut out = vec![vec![0.0f32; n]; n];
    if n == 0 {
        return out;
    }

    for i in 0..n {
        for j in 0..n {
            let i0 = i.saturating_sub(w);
            let j0 = j.saturating_sub(w);
            let i1 = (i + w).min(n - 1);
            let j1 = (j + w).min(n - 1);

            let mut cnt = 0usize;
            let mut le = 0usize;
            let v = s[i][j];

            #[allow(clippy::needless_range_loop)]
            for ii in i0..=i1 {
                for jj in j0..=j1 {
                    cnt += 1;
                    if s[ii][jj] <= v {
                        le += 1;
                    }
                }
            }

            out[i][j] = if cnt > 0 {
                (le as f32) / (cnt as f32)
            } else {
                0.0
            };
        }
    }
    out
}

/// boundary score at position b (between b-1 and b) using diagonal blocks of size `d`.
/// we compare cohesion of the left and right squares to the cross blocks:
/// score = mean(L) + mean(R) - 2 * mean(CROSS), larger => stronger boundary.
fn boundary_scores(r: &[Vec<f32>], d: usize) -> Vec<f32> {
    let n = r.len();
    let mut scores = vec![0.0f32; n];
    if n == 0 || d == 0 {
        return scores;
    }

    #[allow(clippy::needless_range_loop)]
    for b in d..(n - d) {
        let l0 = b - d;
        let l1 = b - 1;
        let r0 = b;
        let r1 = b + d - 1;

        let mean_ll = mean_square(r, l0, l0, l1, l1); // left block on diagonal
        let mean_rr = mean_square(r, r0, r0, r1, r1); // right block on diagonal
        let mean_lr = mean_square(r, l0, r0, l1, r1); // left-right cross
        let mean_rl = mean_square(r, r0, l0, r1, l1); // right-left cross

        let cross = 0.5 * (mean_lr + mean_rl);
        let score = mean_ll + mean_rr - 2.0 * cross;

        // normalize into [0,1] for convenience: rank values are [0,1], this score ~ [-1,1]
        // map via (score+1)/2; you can also skip this if you prefer raw contrast
        scores[b] = (score + 1.0) * 0.5;
    }
    scores
}

/// mean of a rectangular submatrix r[i0..=i1][j0..=j1]
fn mean_square(r: &[Vec<f32>], i0: usize, j0: usize, i1: usize, j1: usize) -> f32 {
    let n = r.len();
    if n == 0 {
        return 0.0;
    }
    let i1 = i1.min(n - 1);
    let j1 = j1.min(n - 1);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    #[allow(clippy::needless_range_loop)]
    for i in i0..=i1 {
        for j in j0..=j1 {
            sum += r[i][j];
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum / (cnt as f32) }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn joined_len(slice: &[String]) -> usize {
    if slice.is_empty() {
        return 0;
    }
    // length of "s0 s1 s2 ..." with single spaces
    let spaces = slice.len().saturating_sub(1);
    slice.iter().map(|s| s.len()).sum::<usize>() + spaces
}

fn sentences_hash(sentences: &[String]) -> String {
    let mut h = blake3::Hasher::new();
    for s in sentences {
        h.update(s.as_bytes());
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
}

fn paragraph_cache_path(params: &GroupParams, sentences_hash: &str) -> PathBuf {
    params
        .cache_dir
        .join("rusty-paragraphizer")
        .join(params.hash())
        .join(format!("{sentences_hash}.txt"))
}

fn try_load_cached_paragraphs(
    params: &GroupParams,
    sentences: &[String],
) -> Result<Option<Vec<String>>> {
    let sh = sentences_hash(sentences);
    let path = paragraph_cache_path(params, &sh);

    if !path.exists() {
        return Ok(None);
    }

    let file = fs::File::open(&path)?;
    let reader = BufReader::new(file);

    let paragraphs: Vec<String> = reader
        .lines()
        .map_while(Result::ok)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(Some(paragraphs))
}

fn persist_paragraphs_cache(
    params: &GroupParams,
    sentences: &[String],
    paragraphs: &[String],
) -> Result<()> {
    let sh = sentences_hash(sentences);
    let path = paragraph_cache_path(params, &sh);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        for p in paragraphs {
            let clean = p.replace('\n', " ").trim().to_string();
            writeln!(f, "{clean}")?;
        }
        f.sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}
