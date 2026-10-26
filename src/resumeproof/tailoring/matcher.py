"""Deterministic hybrid matcher combining keyword/synonym, BM25, and local embeddings via fastembed."""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Optional

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

PINNED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_FASTEMBED_VERSION = "0.9.1"

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
    "api": {"rest", "restful", "fastapi", "endpoints", "backend", "grpc", "json_rpc", "mcp", "rpc"},
    "backend": {"api", "rest", "fastapi", "server", "mcp"},
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
    "vision": {"cv", "detection", "yolo", "yolo11", "inspection", "defect", "pytorch"},
    "yolo": {"yolo11", "detection", "vision", "pytorch"},
    "yolo11": {"yolo", "detection", "vision"},
    "wal": {"write_ahead", "fsync", "o_direct", "disk_sync", "logger"},
    "sync": {"fsync", "flush", "persistence"},
    "graph": {"graphs", "networkx", "topology", "traversal", "nodes"},
    "testing": {"pytest", "benchmark", "ablation", "eval", "tests", "unit_tests"},
    "ablation": {"variant", "baseline", "benchmark", "comparison", "evaluation"},
}


@dataclass
class MatchScore:
    total: float
    keyword_score: float
    bm25_score: float
    embedding_score: float
    embedding_model: str = PINNED_EMBEDDING_MODEL

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "keyword_score": round(self.keyword_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "embedding_score": round(self.embedding_score, 4),
            "embedding_model": self.embedding_model,
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


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors [0, 1]."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, float(dot)))


class HybridMatcher:
    """Hybrid Matcher combining Keyword/Synonym (0.4), BM25 (0.2), and Local Embeddings (0.4).

    Guarantees:
    - 100% deterministic (no random seed, no temperature)
    - Zero LLM calls in scoring
    - Real ONNX sentence embeddings via fastembed (pinned to sentence-transformers/all-MiniLM-L6-v2)
    - Zero-tolerance failure: raises RuntimeError if embeddings return zeros or fail
    - Disk-cached embeddings
    """

    def __init__(self, cache_dir: Optional[Path] = None, model_name: str = PINNED_EMBEDDING_MODEL):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path(".cache/matcher")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_hash = hashlib.sha256(self.model_name.encode("utf-8")).hexdigest()[:8]
        self.cache_file = self.cache_dir / f"embeddings_cache_{model_hash}.json"
        self._disk_cache: dict[str, list[float]] = self._load_cache()
        logger.info(f"Initialized HybridMatcher with fastembed model '{self.model_name}' (v{PINNED_FASTEMBED_VERSION})")
        self._embedder = TextEmbedding(model_name=self.model_name)

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
        """Retrieves or calculates fastembed embedding vector with disk caching.
        Raises RuntimeError immediately if the model returns all zeros or an empty vector."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot embed empty text.")

        cache_key = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        if cache_key in self._disk_cache:
            return self._disk_cache[cache_key]

        embeddings = list(self._embedder.embed([clean_text]))
        if not embeddings:
            raise RuntimeError(f"fastembed ({self.model_name}) produced empty embedding for '{clean_text[:40]}'")

        vec = [float(x) for x in embeddings[0]]
        if all(abs(x) < 1e-9 for x in vec):
            raise RuntimeError(f"fastembed ({self.model_name}) returned all zeros for '{clean_text[:40]}' - silent fallback blocked.")

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        else:
            raise RuntimeError(f"fastembed ({self.model_name}) returned vector with zero L2 norm for '{clean_text[:40]}'")

        self._disk_cache[cache_key] = vec
        self._save_cache()
        return vec

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
            embedding_model=self.model_name,
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
