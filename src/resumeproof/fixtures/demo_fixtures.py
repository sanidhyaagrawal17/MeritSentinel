"""Recorded fixtures for offline tests and CI across the 3 demo personas."""

ALEX_RIVERA_FIXTURE = {
    "profile_facts": [
        {"fact_type": "name", "value": "Alex Rivera", "source_line": 1},
        {"fact_type": "location", "value": "San Francisco, CA", "source_line": 1},
        {"fact_type": "email", "value": "alex.rivera@example.com", "source_line": 1},
        {"fact_type": "phone", "value": "(555) 234-5678", "source_line": 1},
        {"fact_type": "education_institution", "value": "University of California, Berkeley", "source_line": 3},
        {"fact_type": "degree", "value": "B.S. in Computer Science", "source_line": 3},
        {"fact_type": "cgpa", "value": "3.8 / 4.0", "source_line": 3},
        {"fact_type": "graduation_date", "value": "May 2021", "source_line": 3},
        {"fact_type": "employer_name", "value": "CloudScale Systems", "source_line": 5},
        {"fact_type": "employment_dates", "value": "June 2021 - Present", "source_line": 5},
    ],
    "claims": [
        {
            "claim_type": "project_metric",
            "project_id": "raft-kv",
            "text": "Implemented Raft consensus protocol with leader election timeouts of 150-300ms maintaining 99.99% linearizability.",
            "tools": ["Go", "Raft", "gRPC"],
            "metric_values": ["150-300ms", "99.99%"],
            "source_line": 10
        },
        {
            "claim_type": "project_metric",
            "project_id": "raft-kv",
            "text": "Sustained 45,000 req/sec write throughput under 3-node cluster benchmark.",
            "tools": ["Go", "Raft"],
            "metric_values": ["45,000 req/sec"],
            "source_line": 11
        },
        {
            "claim_type": "project_metric",
            "project_id": "raft-kv",
            "text": "Handled 100,000 req/sec write throughput with zero packet loss under split-brain network partition.",
            "tools": ["Go", "Raft"],
            "metric_values": ["100,000 req/sec"],
            "source_line": 12
        },
        {
            "claim_type": "project_metric",
            "project_id": "raft-kv",
            "text": "Built Paxos consensus engine and migrated infrastructure to Kubernetes with Istio service mesh.",
            "tools": ["Paxos", "Kubernetes", "Istio"],
            "metric_values": [],
            "source_line": 13
        },
        {
            "claim_type": "project_metric",
            "project_id": "wal-engine",
            "text": "Engineered crash-resilient write-ahead logger with p99 fsync latency of 1.13ms using O_DIRECT on NVMe.",
            "tools": ["C++", "Python", "POSIX"],
            "metric_values": ["1.13ms"],
            "source_line": 15
        },
        {
            "claim_type": "project_metric",
            "project_id": "flow-limiter",
            "text": "Distributed sliding-window token bucket limiter processing 80,000 req/sec at sub-millisecond p99 latency.",
            "tools": ["Go", "Redis"],
            "metric_values": ["80,000 req/sec"],
            "source_line": 18
        },
        {
            "claim_type": "skill",
            "project_id": None,
            "text": "Distributed systems engineering with Raft consensus, POSIX direct I/O, and Redis token bucket limiting.",
            "tools": ["Go", "C++", "Python", "Redis", "Raft"],
            "metric_values": [],
            "source_line": 20
        }
    ]
}

PRIYA_NAIR_FIXTURE = {
    "profile_facts": [
        {"fact_type": "name", "value": "Priya Nair", "source_line": 1},
        {"fact_type": "location", "value": "New York, NY", "source_line": 1},
        {"fact_type": "email", "value": "priya.nair@example.com", "source_line": 1},
        {"fact_type": "phone", "value": "(555) 987-6543", "source_line": 1},
        {"fact_type": "education_institution", "value": "Columbia University", "source_line": 3},
        {"fact_type": "degree", "value": "M.S. in Data Science", "source_line": 3},
        {"fact_type": "cgpa", "value": "3.9 / 4.0", "source_line": 3},
        {"fact_type": "graduation_date", "value": "December 2022", "source_line": 3},
        {"fact_type": "employer_name", "value": "DataScale Analytics", "source_line": 5},
        {"fact_type": "employment_dates", "value": "January 2023 - Present", "source_line": 5},
    ],
    "claims": [
        {
            "claim_type": "project_metric",
            "project_id": "feature-forge",
            "text": "Streamed feature vectors achieving p95 lookup latency under 4.2ms across 2 million customer keys.",
            "tools": ["Python", "Kafka", "DuckDB"],
            "metric_values": ["4.2ms", "2 million"],
            "source_line": 10
        },
        {
            "claim_type": "project_metric",
            "project_id": "feature-forge",
            "text": "Processed 10 TB/day of streaming telemetry logs with zero buffering lag.",
            "tools": ["Python", "Kafka"],
            "metric_values": ["10 TB/day"],
            "source_line": 11
        },
        {
            "claim_type": "project_metric",
            "project_id": "tabular-boost",
            "text": "Improved classification AUC-ROC from 0.81 to 0.88 using LightGBM feature interaction trees.",
            "tools": ["LightGBM", "Python"],
            "metric_values": ["0.81", "0.88"],
            "source_line": 13
        },
        {
            "claim_type": "project_metric",
            "project_id": "tabular-boost",
            "text": "Accelerated PyTorch model scoring 3.4x using ONNX Runtime graph optimizations.",
            "tools": ["PyTorch", "ONNX Runtime"],
            "metric_values": ["3.4x"],
            "source_line": 14
        },
        {
            "claim_type": "project_metric",
            "project_id": "tabular-boost",
            "text": "Distributed training across 16 GPUs using PyTorch FSDP.",
            "tools": ["PyTorch", "PyTorch FSDP"],
            "metric_values": ["16 GPUs"],
            "source_line": 15
        },
        {
            "claim_type": "project_metric",
            "project_id": "tabular-boost",
            "text": "Trained 70B parameter LLM from scratch using DeepSpeed and fine-tuned custom diffusion models.",
            "tools": ["DeepSpeed"],
            "metric_values": ["70B"],
            "source_line": 16
        },
        {
            "claim_type": "skill",
            "project_id": None,
            "text": "Tabular ML optimization and streaming feature engineering.",
            "tools": ["PyTorch", "LightGBM", "ONNX Runtime", "Kafka", "DuckDB"],
            "metric_values": [],
            "source_line": 18
        }
    ]
}

SAM_OKAFOR_FIXTURE = {
    "profile_facts": [
        {"fact_type": "name", "value": "Sam Okafor", "source_line": 1},
        {"fact_type": "location", "value": "Chicago, IL", "source_line": 1},
        {"fact_type": "email", "value": "sam.okafor@example.com", "source_line": 1},
        {"fact_type": "phone", "value": "(555) 432-1098", "source_line": 1},
        {"fact_type": "education_institution", "value": "Illinois Tech", "source_line": 3},
        {"fact_type": "degree", "value": "B.Tech in Computer Science and Engineering", "source_line": 3},
        {"fact_type": "cgpa", "value": "7.3 / 10.0", "source_line": 3},
        {"fact_type": "graduation_date", "value": "June 2024", "source_line": 3},
    ],
    "claims": [
        {
            "claim_type": "project_metric",
            "project_id": "task-tracker-api",
            "text": "Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
            "tools": ["FastAPI", "SQLite", "Python"],
            "metric_values": ["350 req/sec"],
            "source_line": 6
        },
        {
            "claim_type": "project_metric",
            "project_id": "task-tracker-api",
            "text": "Built enterprise microservices architecture supporting millions of global enterprise users.",
            "tools": ["FastAPI", "Docker"],
            "metric_values": ["millions"],
            "source_line": 8
        },
        {
            "claim_type": "project_metric",
            "project_id": "mini-dns-resolver",
            "text": "Implemented recursive UDP DNS resolver resolving queries in under 45ms for cached records.",
            "tools": ["Python", "Sockets", "DNS"],
            "metric_values": ["45ms"],
            "source_line": 10
        },
        {
            "claim_type": "project_metric",
            "project_id": "mini-dns-resolver",
            "text": "Prevented enterprise-level DDoS attacks using automated BGP route scrubbing.",
            "tools": ["BGP"],
            "metric_values": [],
            "source_line": 12
        },
        {
            "claim_type": "project_metric",
            "project_id": "packet-sniffer",
            "text": "Captured and decoded IPv4, TCP, and UDP packet headers with custom bitwise unpacking.",
            "tools": ["Python", "Sockets", "TCP/IP"],
            "metric_values": [],
            "source_line": 14
        },
        {
            "claim_type": "project_metric",
            "project_id": "packet-sniffer",
            "text": "Architected Apache Kafka event bus for financial transaction settlement.",
            "tools": ["Kafka"],
            "metric_values": [],
            "source_line": 15
        },
        {
            "claim_type": "project_metric",
            "project_id": "packet-sniffer",
            "text": "Designed distributed Raft consensus cluster with Byzantine fault tolerance.",
            "tools": ["Raft"],
            "metric_values": [],
            "source_line": 16
        },
        {
            "claim_type": "skill",
            "project_id": None,
            "text": "Python networking programming, socket handling, and FastAPI development.",
            "tools": ["Python", "FastAPI", "SQLite", "TCP/IP"],
            "metric_values": [],
            "source_line": 18
        }
    ]
}

DEMO_FIXTURES = {
    "alex_rivera": ALEX_RIVERA_FIXTURE,
    "priya_nair": PRIYA_NAIR_FIXTURE,
    "sam_okafor": SAM_OKAFOR_FIXTURE,
}
