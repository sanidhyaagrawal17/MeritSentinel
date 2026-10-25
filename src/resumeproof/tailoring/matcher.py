"""Deterministic hybrid matcher combining keyword/synonym, BM25, and local embeddings."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional

# Standard stopwords for matching
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "or", "our", "we", "you", "your",
    "must", "should", "experience", "knowledge", "ability", "proficient",
    "years", "working", "strong", "understanding", "plus", "required",
    "low", "high", "good", "great", "excellent"
}

# Technical synonym expansions
SYNONYMS = {
    "latency": {"p99", "p95", "response_time", "delay", "sub_millisecond", "ms"},
    "throughput": {"req_sec", "rps", "qps", "req/sec", "requests", "reqs"},
    "database": {"db", "sql", "sqlite", "postgres", "postgresql", "mysql", "duckdb"},
    "api": {"rest", "restful", "fastapi", "endpoints", "backend", "grpc"},
    "caching": {"cache", "redis", "ttl", "in-memory", "cached"},
    "consensus": {"raft", "paxos", "replication", "leader", "election", "linearizable"},
    "concurrency": {"threads", "concurrent", "parallel", "locks", "pooling", "pool"},
    "network": {"networking", "sockets", "udp", "tcp", "ip", "ipv4", "raw_sockets"},
    "go": {"golang"},
    "golang": {"go"},
    "python": {"py", "fastapi"},
    "k8s": {"kubernetes"},
    "kubernetes": {"k8s"},
    "ml": {"machine_learning", "gbdt", "lightgbm", "onnx", "model", "inference", "tabular"},
}


@dataclass
class MatchScore:
    total: float
    keyword_score: float
    bm25_score: float
    embedding_score: float

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "keyword_score": round(self.keyword_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "embedding_score": round(self.embedding_score, 4),
        }


def _tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", text) if len(t) >= 2]


def _expand_tokens_with_synonyms(tokens: list[str]) -> set[str]:
    """Expands token list with technical synonyms."""
    expanded = set()
    for t in tokens:
        expanded.add(t)
        if t in SYNONYMS:
            expanded.update(SYNONYMS[t])
    return expanded


def _compute_deterministic_embedding(text: str, dim: int = 128) -> list[float]:
    """Computes a deterministic, fixed-dimension feature vector from subwords and character n-grams."""
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector

    # Word-level features and character 3-grams
    features = list(tokens)
    for t in tokens:
        if len(t) >= 3:
            for i in range(len(t) - 2):
                features.append(t[i:i+3])

    for feat in features:
        h = int(hashlib.sha256(feat.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vector[idx] += sign

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0.0:
        vector = [x / norm for x in vector]
    return vector


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))


class HybridMatcher:
    """Hybrid Matcher combining Keyword/Synonym (0.4), BM25 (0.2), and Local Embeddings (0.4).

    Guarantees:
    - 100% deterministic (no random seed, no temperature)
    - Zero LLM calls in scoring
    - Disk-cached embeddings
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".cache/matcher")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.json"
        self._disk_cache: dict[str, list[float]] = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._disk_cache), encoding="utf-8")
        except Exception:
            pass

    def get_embedding(self, text: str) -> list[float]:
        """Retrieves or calculates deterministic embedding vector with disk caching."""
        cache_key = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if cache_key in self._disk_cache:
            return self._disk_cache[cache_key]

        emb = _compute_deterministic_embedding(text)
        self._disk_cache[cache_key] = emb
        self._save_cache()
        return emb

    def compute_keyword_score(self, query: str, document: str) -> float:
        """Computes keyword and technical synonym overlap score (0.0 to 1.0)."""
        q_tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
        if not q_tokens:
            return 0.0

        q_expanded = _expand_tokens_with_synonyms(q_tokens)
        doc_tokens = set(_tokenize(document))
        doc_expanded = _expand_tokens_with_synonyms(list(doc_tokens))

        matches = q_expanded.intersection(doc_expanded)
        # Ratio of matched query terms
        direct_matches = sum(1 for t in q_tokens if t in doc_expanded or any(syn in doc_expanded for syn in SYNONYMS.get(t, set())))
        ratio = direct_matches / len(q_tokens)

        # Bonus for exact string containment
        if query.strip().lower() in document.strip().lower():
            ratio = 1.0

        return max(0.0, min(1.0, ratio))

    def compute_bm25_scores(self, query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        """Computes BM25 Okapi scores across candidate documents normalized to [0, 1]."""
        if not documents:
            return []

        tokenized_docs = [_tokenize(d) for d in documents]
        tokenized_query = [t for t in _tokenize(query) if t not in STOPWORDS]

        N = len(documents)
        avgdl = sum(len(d) for d in tokenized_docs) / max(1, N)

        # Document frequencies
        df: dict[str, int] = {}
        for d in tokenized_docs:
            seen = set(d)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        raw_scores = []
        for d in tokenized_docs:
            score = 0.0
            doc_len = len(d)
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1

            for qt in tokenized_query:
                if qt in df:
                    doc_freq = df[qt]
                    idf = math.log(1.0 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
                    tf = counts.get(qt, 0)
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                    score += idf * (numerator / max(1e-5, denominator))
            raw_scores.append(max(0.0, score))

        # Normalize raw scores to [0, 1]
        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score > 0.0:
            return [min(1.0, s / max_score) for s in raw_scores]
        return [0.0] * N

    def compute_embedding_score(self, query: str, document: str) -> float:
        """Computes cosine similarity between query and document embeddings."""
        v1 = self.get_embedding(query)
        v2 = self.get_embedding(document)
        return _cosine_similarity(v1, v2)

    def match(self, query: str, document: str, bm25_normalized_score: float = 0.0) -> MatchScore:
        """Computes hybrid match score for a single document: 0.4 keyword + 0.2 BM25 + 0.4 embedding."""
        kw = self.compute_keyword_score(query, document)
        emb = self.compute_embedding_score(query, document)
        bm25 = max(0.0, min(1.0, bm25_normalized_score))

        total = (0.4 * kw) + (0.2 * bm25) + (0.4 * emb)
        return MatchScore(
            total=total,
            keyword_score=kw,
            bm25_score=bm25,
            embedding_score=emb,
        )

    def rank_candidates(self, query: str, candidate_texts: list[str]) -> list[tuple[int, MatchScore]]:
        """Ranks candidate texts against query using the full hybrid pipeline."""
        if not candidate_texts:
            return []

        bm25_scores = self.compute_bm25_scores(query, candidate_texts)
        results = []
        for idx, doc in enumerate(candidate_texts):
            score = self.match(query, doc, bm25_normalized_score=bm25_scores[idx])
            results.append((idx, score))

        results.sort(key=lambda x: x[1].total, reverse=True)
        return results
