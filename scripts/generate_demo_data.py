"""Script to generate realistic test personas and demo data for ResumeProof."""

from pathlib import Path
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DEMO_DIR = Path("data/demo")

def build_pdf_resume(filepath: Path, title: str, text_lines: list[str]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=3
    )

    story = [Paragraph(f"<b>{title}</b>", header_style)]
    for line in text_lines:
        line_clean = line.strip()
        if not line_clean:
            story.append(Spacer(1, 4))
        elif line_clean.startswith("## "):
            story.append(Paragraph(f"<b>{line_clean[3:].upper()}</b>", section_style))
        elif line_clean.startswith("# "):
            pass # title handled
        else:
            story.append(Paragraph(line_clean.replace("&", "&amp;"), body_style))

    doc.build(story)

def create_alex_rivera():
    alex_dir = DEMO_DIR / "alex_rivera"
    alex_dir.mkdir(parents=True, exist_ok=True)

    resume_lines = [
        "Alex Rivera | San Francisco, CA | alex.rivera@example.com | (555) 234-5678",
        "## Education",
        "B.S. in Computer Science, University of California, Berkeley (CGPA: 3.8 / 4.0, Graduated May 2021)",
        "## Experience",
        "Senior Infrastructure Engineer, CloudScale Systems (June 2021 - Present)",
        "- Designed distributed replication topology across 3 geographic zones.",
        "- Implemented low-latency write ahead logs with disk sync.",
        "## Projects",
        "Raft-KV Distributed Store (Go, Raft, gRPC):",
        "- Implemented Raft consensus protocol with leader election timeouts of 150-300ms maintaining 99.99% linearizability.",
        "- Sustained 45,000 req/sec write throughput under 3-node cluster benchmark.",
        "- Claim: Handled 100,000 req/sec write throughput with zero packet loss under split-brain network partition.",
        "- Claim: Built Paxos consensus engine and migrated infrastructure to Kubernetes with Istio service mesh.",
        "WAL-Engine (C++, Python, POSIX):",
        "- Engineered crash-resilient write-ahead logger with p99 fsync latency of 1.13ms using O_DIRECT on NVMe.",
        "- Achieved zero byte corruption across 500 simulated sudden kernel panic injection cycles.",
        "Flow-Limiter (Go, Redis):",
        "- Distributed sliding-window token bucket limiter processing 80,000 req/sec at sub-millisecond p99 latency.",
        "## Skills",
        "Languages: Go, C++, Python, SQL",
        "Distributed Systems: Raft, gRPC, Redis, POSIX I/O, Consensus protocols",
        "Tools: Docker, Git, Linux, Prometheus"
    ]

    resume_md = "# Alex Rivera\n" + "\n".join(resume_lines)
    (alex_dir / "resume.md").write_text(resume_md, encoding="utf-8")
    build_pdf_resume(alex_dir / "resume.pdf", "Alex Rivera", resume_lines)

    # Projects
    p1 = alex_dir / "projects" / "raft-kv"
    p1.mkdir(parents=True, exist_ok=True)
    p1_readme = """# Raft-KV Distributed Key-Value Store
A replicated distributed key-value store in Go implementing the Raft consensus algorithm.

## Features
- Raft leader election with randomized timeouts (150ms - 300ms).
- AppendEntries log replication and state machine application.
- Snapshotting and log compaction.

## Verified Benchmarks
- Tested on 3 nodes (AWS c5.xlarge).
- Max stable write throughput: 45,000 req/sec.
- 99.99% linearizable reads under concurrent Jepsen partition tests.
- Note: Never tested above 3 nodes or 45k req/sec. No Paxos engine.
"""
    (p1 / "README.md").write_text(p1_readme, encoding="utf-8")
    (p1 / "design_notes.md").write_text("Leader election uses Raft paper Section 5.2. Heartbeats every 50ms.\nNetwork partition recovery tested via simulated dropped TCP connections.", encoding="utf-8")
    (p1 / "metrics.json").write_text(json.dumps({"max_req_sec": 45000, "election_timeout_ms": [150, 300], "linearizability": 0.9999, "cluster_size": 3}), encoding="utf-8")

    p2 = alex_dir / "projects" / "wal-engine"
    p2.mkdir(parents=True, exist_ok=True)
    p2_readme = """# WAL-Engine
High-performance Write-Ahead Logging storage engine in C++ with Python bindings.

## Architecture
- Direct I/O (O_DIRECT) bypasses OS page cache to eliminate double-buffering.
- 4KB aligned circular ring buffer for sequential commits.

## Verified Benchmarks
- NVMe p99 fsync latency: 1.13ms under 10,000 concurrent commits/sec.
- Zero corrupt records detected over 500 sudden kill -9 / power cut fault injections.
"""
    (p2 / "README.md").write_text(p2_readme, encoding="utf-8")
    (p2 / "design_notes.md").write_text("Uses crc32 IEEE checksum on log headers. Segment files rolled at 64MB.\nCrash recovery validates CRC on replay and truncates torn writes.", encoding="utf-8")
    (p2 / "metrics.json").write_text(json.dumps({"fsync_latency_p99_ms": 1.13, "fault_injection_cycles": 500, "corrupt_bytes": 0}), encoding="utf-8")

    p3 = alex_dir / "projects" / "flow-limiter"
    p3.mkdir(parents=True, exist_ok=True)
    p3_readme = """# Flow-Limiter
Distributed sliding-window rate limiter in Go with Redis Lua scripts.

## Performance
- Processes 80,000 req/sec with p99 latency < 0.8ms.
- Atomic Lua script evaluates token count and timestamps in single round-trip.
"""
    (p3 / "README.md").write_text(p3_readme, encoding="utf-8")
    (p3 / "design_notes.md").write_text("Sliding log algorithm prevents burst boundary spikes.\nFallback local in-memory leaky bucket activates if Redis heartbeat drops.", encoding="utf-8")
    (p3 / "metrics.json").write_text(json.dumps({"req_sec": 80000, "p99_latency_ms": 0.8}), encoding="utf-8")

    # Traps & Evaluation Answers
    traps = {
        "supported": [
            {"claim": "Raft leader election election timeout of 150-300ms maintaining 99.99% linearizability", "category": "project_metric"},
            {"claim": "Engineered crash-resilient write-ahead logger with p99 fsync latency of 1.13ms using O_DIRECT", "category": "project_metric"}
        ],
        "partial": [
            {"claim": "Handled 100,000 req/sec write throughput under benchmark", "ground_truth": "Actual verified maximum was 45,000 req/sec."},
            {"claim": "Zero packet loss under split-brain network partition across any cluster size", "ground_truth": "Only verified on 3-node cluster; packet loss occurs during uncommitted leader partition."}
        ],
        "unsupported": [
            {"claim": "Built Paxos consensus engine", "ground_truth": "Never implemented Paxos; only Raft was used."},
            {"claim": "Migrated infrastructure to Kubernetes with Istio service mesh", "ground_truth": "No Kubernetes or Istio in any project doc or code."}
        ]
    }
    (alex_dir / "planted_traps.json").write_text(json.dumps(traps, indent=2), encoding="utf-8")

    answers = {
        "strong": "In Raft-KV, we configure randomized election timeouts between 150ms and 300ms to prevent split votes. When a follower stops receiving heartbeats from the leader within this window, it transitions to candidate, increments currentTerm, votes for itself, and sends RequestVote RPCs. If it gathers votes from a majority (2 out of 3 nodes), it becomes leader.",
        "terse": "Timeouts are 150-300ms randomized. Majority votes win leader role.",
        "confidently_wrong": "In our Paxos consensus engine, the leader leases are managed by Kubernetes etcd and Istio ingress routers, which guarantee 100,000 req/sec write throughput across 50 nodes without election delays.",
        "generic": "Distributed consensus is very important in modern cloud computing. We used standard industry practices to ensure high availability and robust data replication."
    }
    (alex_dir / "sample_answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    jd = """# Senior Distributed Systems Engineer - Infrastructure
## Responsibilities
- Architect high-throughput distributed consensus storage engines.
- Optimize disk I/O, write-ahead logging, and sub-millisecond p99 latencies.
## Required Skills
- Go or C++ systems programming
- Distributed systems (Raft, Paxos, consensus)
- POSIX I/O, O_DIRECT, low-latency storage
## Qualifications
- B.S. or higher in Computer Science
- Proven experience with benchmarking high-throughput systems
"""
    (alex_dir / "test_jd.md").write_text(jd, encoding="utf-8")

def create_priya_nair():
    priya_dir = DEMO_DIR / "priya_nair"
    priya_dir.mkdir(parents=True, exist_ok=True)

    resume_lines = [
        "Priya Nair | New York, NY | priya.nair@example.com | (555) 987-6543",
        "## Education",
        "M.S. in Data Science, Columbia University (CGPA: 3.9 / 4.0, Graduated December 2022)",
        "## Experience",
        "Machine Learning Engineer, DataScale Analytics (January 2023 - Present)",
        "- Deployed real-time tabular feature pipelines with low inference latency.",
        "- Tuned GBDT and ONNX runtime inference engines.",
        "## Projects",
        "Feature-Forge Real-Time Feature Store (Python, Kafka, DuckDB):",
        "- Streamed feature vectors achieving p95 lookup latency under 4.2ms across 2 million customer keys.",
        "- Claim: Processed 10 TB/day of streaming telemetry logs with zero buffering lag.",
        "Tabular-Boost Inference Optimizer (PyTorch, ONNX, LightGBM):",
        "- Improved classification AUC-ROC from 0.81 to 0.88 using LightGBM feature interaction trees.",
        "- Accelerated PyTorch model scoring 3.4x using ONNX Runtime graph optimizations.",
        "- Claim: Distributed training across 16 GPUs using PyTorch FSDP.",
        "- Claim: Trained 70B parameter LLM from scratch using DeepSpeed and fine-tuned custom diffusion models.",
        "## Skills",
        "Languages: Python, SQL",
        "ML & Data: PyTorch, LightGBM, ONNX Runtime, Kafka, DuckDB, Pandas, NumPy",
        "Tools: Docker, Git, MLflow"
    ]

    resume_md = "# Priya Nair\n" + "\n".join(resume_lines)
    (priya_dir / "resume.md").write_text(resume_md, encoding="utf-8")
    build_pdf_resume(priya_dir / "resume.pdf", "Priya Nair", resume_lines)

    p1 = priya_dir / "projects" / "feature-forge"
    p1.mkdir(parents=True, exist_ok=True)
    p1_readme = """# Feature-Forge
Real-time streaming feature store combining Kafka and DuckDB.

## Verified Benchmarks
- Sustained lookup p95 latency under 4.2ms for 2M keys.
- Stream volume verified at 2.5 TB/day in production cluster.
- Note: Did not process 10 TB/day.
"""
    (p1 / "README.md").write_text(p1_readme, encoding="utf-8")
    (p1 / "design_notes.md").write_text("Feature schema enforced with Pydantic. Micro-batches committed to DuckDB columnar storage every 500ms.", encoding="utf-8")
    (p1 / "metrics.json").write_text(json.dumps({"p95_lookup_ms": 4.2, "verified_daily_volume_tb": 2.5, "num_keys": 2000000}), encoding="utf-8")

    p2 = priya_dir / "projects" / "tabular-boost"
    p2.mkdir(parents=True, exist_ok=True)
    p2_readme = """# Tabular-Boost
Inference optimization framework for tabular models.

## Verified Results
- Baseline AUC-ROC 0.81 increased to 0.88 with LightGBM interaction depth 6.
- 3.4x inference speedup using ONNX Runtime FP32 CPU graph execution.
- Tested on 2x NVIDIA T4 GPUs; no 16-GPU FSDP or 70B LLM training.
"""
    (p2 / "README.md").write_text(p2_readme, encoding="utf-8")
    (p2 / "design_notes.md").write_text("ONNX export fused redundant normalization layers. Quantization evaluated but left at FP32 to avoid AUC regression.", encoding="utf-8")
    (p2 / "metrics.json").write_text(json.dumps({"auc_baseline": 0.81, "auc_final": 0.88, "inference_speedup_x": 3.4, "gpus_tested": 2}), encoding="utf-8")

    traps = {
        "supported": [
            {"claim": "Streamed feature vectors achieving p95 lookup latency under 4.2ms across 2 million customer keys", "category": "project_metric"},
            {"claim": "Improved classification AUC-ROC from 0.81 to 0.88 using LightGBM", "category": "project_metric"}
        ],
        "partial": [
            {"claim": "Processed 10 TB/day of streaming telemetry logs", "ground_truth": "Actual verified volume was 2.5 TB/day."},
            {"claim": "Distributed training across 16 GPUs using PyTorch FSDP", "ground_truth": "Only tested on 2 GPUs locally."}
        ],
        "unsupported": [
            {"claim": "Trained 70B parameter LLM from scratch using DeepSpeed", "ground_truth": "No LLM training; project was strictly tabular LightGBM and PyTorch."},
            {"claim": "Fine-tuned custom diffusion models for image synthesis", "ground_truth": "No diffusion models or computer vision in portfolio."}
        ]
    }
    (priya_dir / "planted_traps.json").write_text(json.dumps(traps, indent=2), encoding="utf-8")

    answers = {
        "strong": "In Feature-Forge, we hit a p95 lookup latency under 4.2ms by keeping the active 2 million feature keys indexed in memory with an LRU cache, while DuckDB handled cold lookups via vectorized parquet scans. Kafka consumers partitioned events by user_id to ensure cache locality.",
        "terse": "Used in-memory indexing with DuckDB columnar backing to keep p95 latency at 4.2ms.",
        "confidently_wrong": "We used DeepSpeed ZeRO-3 on 16 GPUs to pretrain our 70B transformer, which directly generated tabular feature embeddings from 10 TB/day of Kafka streams.",
        "generic": "Feature stores are essential for machine learning operations. We optimized pipelines using industry standard frameworks to achieve high throughput."
    }
    (priya_dir / "sample_answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    jd = """# Machine Learning Infrastructure Engineer
## Responsibilities
- Build and maintain low-latency feature stores and model inference servers.
- Optimize tabular model pipelines (LightGBM, PyTorch, ONNX).
## Required Skills
- Python, PyTorch, LightGBM, ONNX Runtime
- Streaming data (Kafka)
- Low-latency inference optimization
"""
    (priya_dir / "test_jd.md").write_text(jd, encoding="utf-8")

def create_sam_okafor():
    sam_dir = DEMO_DIR / "sam_okafor"
    sam_dir.mkdir(parents=True, exist_ok=True)

    resume_lines = [
        "Sam Okafor | Chicago, IL | sam.okafor@example.com | (555) 432-1098",
        "## Education",
        "B.Tech in Computer Science and Engineering, Illinois Tech (CGPA: 7.3 / 10.0, Graduated June 2024)",
        "## Projects",
        "Task-Tracker API (Python, FastAPI, SQLite):",
        "- Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
        "- Implemented JWT authentication and role-based access control.",
        "- Claim: Built enterprise microservices architecture supporting millions of global enterprise users.",
        "Mini-DNS-Resolver (Python, Sockets):",
        "- Implemented recursive UDP DNS resolver resolving queries in under 45ms for cached records.",
        "- Parsed RFC 1035 wire format A and CNAME record headers.",
        "- Claim: Prevented enterprise-level DDoS attacks using automated BGP route scrubbing.",
        "Packet-Sniffer (Python, Raw Sockets):",
        "- Captured and decoded IPv4, TCP, and UDP packet headers with custom bitwise unpacking.",
        "- Claim: Architected Apache Kafka event bus for financial transaction settlement.",
        "- Claim: Designed distributed Raft consensus cluster with Byzantine fault tolerance.",
        "## Skills",
        "Languages: Python, SQL",
        "Networking & Backend: FastAPI, SQLite, Sockets, TCP/IP, UDP, HTTP, DNS",
        "Tools: Git, Linux, Docker"
    ]

    resume_md = "# Sam Okafor\n" + "\n".join(resume_lines)
    (sam_dir / "resume.md").write_text(resume_md, encoding="utf-8")
    build_pdf_resume(sam_dir / "resume.pdf", "Sam Okafor", resume_lines)

    p1 = sam_dir / "projects" / "task-tracker-api"
    p1.mkdir(parents=True, exist_ok=True)
    p1_readme = """# Task-Tracker API
FastAPI task management backend using SQLite.

## Verified Benchmarks
- Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.
- Single node service, not microservices.
"""
    (p1 / "README.md").write_text(p1_readme, encoding="utf-8")
    (p1 / "design_notes.md").write_text("Uses SQLite WAL mode with SQLAlchemy async engine and pool_size=10.", encoding="utf-8")
    (p1 / "metrics.json").write_text(json.dumps({"concurrent_read_req_sec": 350, "db": "SQLite"}), encoding="utf-8")

    p2 = sam_dir / "projects" / "mini-dns-resolver"
    p2.mkdir(parents=True, exist_ok=True)
    p2_readme = """# Mini-DNS-Resolver
A recursive DNS query resolver built with Python raw sockets.

## Verified Results
- Resolves cached DNS queries in under 45ms.
- Parses A and CNAME records per RFC 1035.
- Local toy project; no BGP scrubbing or enterprise DDoS protection.
"""
    (p2 / "README.md").write_text(p2_readme, encoding="utf-8")
    (p2 / "design_notes.md").write_text("Uses Python struct module for unpacking UDP payloads. In-memory dict cache with TTL expiration.", encoding="utf-8")
    (p2 / "metrics.json").write_text(json.dumps({"cached_resolution_ms": 45, "records_supported": ["A", "CNAME"]}), encoding="utf-8")

    p3 = sam_dir / "projects" / "packet-sniffer"
    p3.mkdir(parents=True, exist_ok=True)
    p3_readme = """# Packet-Sniffer
Python packet capture tool using AF_INET raw sockets.

## Verified Features
- Unpacks 20-byte IPv4 headers and extracts protocol (TCP/UDP/ICMP).
- Educational script; no Kafka or Raft consensus.
"""
    (p3 / "README.md").write_text(p3_readme, encoding="utf-8")
    (p3 / "design_notes.md").write_text("Extracts IP version, IHL, TTL, and source/dest IPs using bit shift operations.", encoding="utf-8")
    (p3 / "metrics.json").write_text(json.dumps({"protocols": ["IPv4", "TCP", "UDP"]}), encoding="utf-8")

    traps = {
        "supported": [
            {"claim": "FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec", "category": "project_metric"},
            {"claim": "Recursive UDP DNS resolver resolving queries in under 45ms for cached records", "category": "project_metric"}
        ],
        "partial": [
            {"claim": "Built enterprise microservices architecture supporting millions of global enterprise users", "ground_truth": "Only a single small FastAPI app with SQLite."},
            {"claim": "Prevented enterprise-level DDoS attacks using automated BGP route scrubbing", "ground_truth": "No BGP or DDoS prevention; just a local DNS script."}
        ],
        "unsupported": [
            {"claim": "Architected Apache Kafka event bus for financial transaction settlement", "ground_truth": "No Kafka or financial transactions in project."},
            {"claim": "Designed distributed Raft consensus cluster with Byzantine fault tolerance", "ground_truth": "No Raft or Byzantine consensus; junior student project."}
        ]
    }
    (sam_dir / "planted_traps.json").write_text(json.dumps(traps, indent=2), encoding="utf-8")

    answers = {
        "strong": "In Task-Tracker API, SQLite can become a write bottleneck, so we enabled WAL (Write-Ahead Logging) mode and tuned SQLAlchemy connection pooling to 10 connections. For read-heavy operations, this allowed concurrent reads while WAL checkpointing happened in the background, reaching 350 read req/sec in our locust test.",
        "terse": "Enabled WAL mode and SQLAlchemy connection pool of 10 to hit 350 req/sec.",
        "confidently_wrong": "We used an Apache Kafka event bus and BGP scrubbing centers to stream financial transactions into our Byzantine fault-tolerant Raft cluster with zero downtime.",
        "generic": "APIs require good database connections and error handling. We followed standard tutorials to make sure the app responded quickly."
    }
    (sam_dir / "sample_answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    jd = """# Junior Backend / Network Software Engineer
## Responsibilities
- Develop Python APIs and networking utilities.
- Maintain relational database connections (SQLite/PostgreSQL).
- Understand TCP/IP, UDP, DNS protocols.
## Qualifications
- B.Tech or B.S. in Computer Science
- Basic Python backend programming and networking fundamentals
"""
    (sam_dir / "test_jd.md").write_text(jd, encoding="utf-8")

if __name__ == "__main__":
    create_alex_rivera()
    create_priya_nair()
    create_sam_okafor()
    print("Successfully generated all demo data and PDF resumes.")
