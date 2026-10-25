"""Tests for deterministic HybridMatcher (Keyword/Synonym 0.4, BM25 0.2, Embedding 0.4)."""

import pytest
from pathlib import Path
from resumeproof.tailoring.matcher import HybridMatcher, MatchScore, SYNONYMS

def test_matcher_weights_and_deterministic_scoring(tmp_path: Path):
    """Verifies that the hybrid matcher adheres strictly to 0.4 keyword + 0.2 BM25 + 0.4 embedding."""
    matcher = HybridMatcher(cache_dir=tmp_path / "cache")
    query = "Distributed rate limiting with sub-millisecond p99 latency in Go"
    doc1 = "Built distributed token bucket rate limiter in Go sustaining 80,000 req/sec with p99 latency under 0.8ms."
    doc2 = "Created Python web scraper for news articles storing results in SQLite database."

    score1 = matcher.match(query, doc1, bm25_normalized_score=1.0)
    score2 = matcher.match(query, doc2, bm25_normalized_score=0.1)

    # Invariant: weights must sum to 1.0 (0.4 + 0.2 + 0.4)
    expected_total1 = (0.4 * score1.keyword_score) + (0.2 * 1.0) + (0.4 * score1.embedding_score)
    assert abs(score1.total - expected_total1) < 1e-6
    assert score1.total > score2.total

    # Invariant: completely deterministic on repeated runs
    score1_repeat = matcher.match(query, doc1, bm25_normalized_score=1.0)
    assert score1.total == score1_repeat.total
    assert score1.keyword_score == score1_repeat.keyword_score
    assert score1.embedding_score == score1_repeat.embedding_score

def test_synonym_expansion(tmp_path: Path):
    """Verifies that technical synonyms (e.g. latency -> p99, throughput -> req/sec) are recognized."""
    matcher = HybridMatcher(cache_dir=tmp_path / "cache")
    query = "Low latency high throughput API"
    doc = "Sustained 350 req/sec with sub-millisecond p99 response times on RESTful endpoints."

    kw_score = matcher.compute_keyword_score(query, doc)
    assert kw_score >= 0.5, f"Expected synonym matching to score >= 0.5, got {kw_score}"

def test_bm25_scoring_and_normalization(tmp_path: Path):
    """Verifies BM25 Okapi scoring across a corpus of documents."""
    matcher = HybridMatcher(cache_dir=tmp_path / "cache")
    corpus = [
        "Distributed Raft consensus algorithm with leader election timeouts in Go",
        "SQLite connection pooling in FastAPI backend",
        "Packet capture sniffer with raw AF_INET sockets",
    ]
    query = "Raft consensus leader election"
    scores = matcher.compute_bm25_scores(query, corpus)

    assert len(scores) == 3
    assert scores[0] == 1.0  # Top match normalized to 1.0
    assert scores[1] < 0.2
    assert scores[2] < 0.2

def test_rank_candidates(tmp_path: Path):
    """Verifies candidate ranking across multiple documents."""
    matcher = HybridMatcher(cache_dir=tmp_path / "cache")
    candidates = [
        "Streamed feature vectors with p95 lookup latency under 4.2ms in DuckDB",
        "Raft consensus engine with 45,000 req/sec write throughput",
        "SQLite WAL mode sustaining 350 concurrent read req/sec",
    ]
    query = "Raft consensus key-value store benchmark"
    ranked = matcher.rank_candidates(query, candidates)

    assert len(ranked) == 3
    top_idx, top_score = ranked[0]
    assert top_idx == 1  # Raft candidate
    assert top_score.total > ranked[1][1].total
