"""Generate 50 calibration items for human labeling and evaluation."""

import csv
from pathlib import Path

ITEMS = [
    # --- TASK-TRACKER-API (8 items) ---
    {
        "id": "val_01",
        "project": "task-tracker-api",
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "answer": "We configured SQLite with WAL (Write-Ahead Logging) mode and tuned the SQLAlchemy async connection pool to pool_size=10. This decoupled concurrent readers from WAL background checkpointing, allowing 350 read req/sec in Locust benchmarking.",
        "intended_tier": "4"
    },
    {
        "id": "val_02",
        "project": "task-tracker-api",
        "question": "How did you scale SQLite to handle 350 concurrent read requests per second?",
        "answer": "We enabled WAL mode and used SQLAlchemy connection pooling set to 10 connections to achieve 350 req/sec in load tests.",
        "intended_tier": "3"
    },
    {
        "id": "val_03",
        "project": "task-tracker-api",
        "question": "What database architecture was used for Task-Tracker API?",
        "answer": "We used SQLite with connection pooling to hit 350 requests per second.",
        "intended_tier": "2"
    },
    {
        "id": "val_04",
        "project": "task-tracker-api",
        "question": "How did you optimize API database queries?",
        "answer": "We indexed tables and made sure our FastAPI endpoints handled errors asynchronously.",
        "intended_tier": "1"
    },
    {
        "id": "val_05",
        "project": "task-tracker-api",
        "question": "How does your task backend handle distributed consensus across multi-region instances?",
        "answer": "We deployed a globally replicated Apache Cassandra cluster with Raft consensus across 5 AWS regions to guarantee zero-downtime multi-master replication.",
        "intended_tier": "0"
    },
    {
        "id": "val_06",
        "project": "task-tracker-api",
        "question": "Did you implement a microservices architecture for Task-Tracker API?",
        "answer": "No, it was built as a single-node FastAPI service with an embedded SQLite database, not microservices.",
        "intended_tier": "3"
    },
    {
        "id": "val_07",
        "project": "task-tracker-api",
        "question": "How does SQLite WAL mode handle concurrent writes and checkpointing in your system?",
        "answer": "In WAL mode, writes append sequentially to the -wal file rather than overwriting database pages directly. Readers read without locking out writers. When the WAL hits a threshold, passive checkpointing merges pages back to the main DB file while writers briefly synchronize.",
        "intended_tier": "4"
    },
    {
        "id": "val_08",
        "project": "task-tracker-api",
        "question": "What ORM connection pool parameters did you configure?",
        "answer": "SQLAlchemy pool_size=10 with async SQLite engine.",
        "intended_tier": "3"
    },

    # --- MINI-DNS-RESOLVER (7 items) ---
    {
        "id": "val_09",
        "project": "mini-dns-resolver",
        "question": "How does your resolver unpack and parse DNS wire-format messages?",
        "answer": "We used Python's struct module to unpack the 12-byte DNS header (Transaction ID, Flags, QDCOUNT, ANCOUNT). We then parse labels iteratively using length-prefixed bytes according to RFC 1035, supporting A and CNAME records.",
        "intended_tier": "4"
    },
    {
        "id": "val_10",
        "project": "mini-dns-resolver",
        "question": "What DNS records does your mini-resolver support?",
        "answer": "It parses standard A records and CNAME records as specified in RFC 1035.",
        "intended_tier": "3"
    },
    {
        "id": "val_11",
        "project": "mini-dns-resolver",
        "question": "How fast does cached DNS resolution respond?",
        "answer": "Cached queries resolve in under 45ms using an in-memory dictionary cache.",
        "intended_tier": "3"
    },
    {
        "id": "val_12",
        "project": "mini-dns-resolver",
        "question": "How is caching implemented in your DNS resolver?",
        "answer": "We stored resolved IP records in Python memory.",
        "intended_tier": "2"
    },
    {
        "id": "val_13",
        "project": "mini-dns-resolver",
        "question": "How did you protect the resolver against multi-gigabit DDoS amplification attacks?",
        "answer": "We configured Cloudflare Magic Transit and BGP Anycast route scrubbers with automated eBPF XDP rate limiting to mitigate 100 Gbps amplification attacks.",
        "intended_tier": "0"
    },
    {
        "id": "val_14",
        "project": "mini-dns-resolver",
        "question": "How does your resolver handle recursive resolution when a CNAME is returned?",
        "answer": "When an answer section contains a CNAME record, the resolver extracts the canonical domain name and initiates a subsequent recursive query for that canonical name until an A record is resolved or recursion depth is exceeded.",
        "intended_tier": "4"
    },
    {
        "id": "val_15",
        "project": "mini-dns-resolver",
        "question": "What socket library did you use for UDP query handling?",
        "answer": "Standard Python socket library with AF_INET and SOCK_DGRAM.",
        "intended_tier": "3"
    },

    # --- PACKET-SNIFFER (6 items) ---
    {
        "id": "val_16",
        "project": "packet-sniffer",
        "question": "How does your packet sniffer unpack IPv4 packet headers?",
        "answer": "We open a raw socket using socket(AF_INET, SOCK_RAW, IPPROTO_IP), unpack the first 20 bytes with struct.unpack('!BBHHHBBH4s4s'), extract the IHL and protocol field, and convert source and destination IP addresses using inet_ntoa.",
        "intended_tier": "4"
    },
    {
        "id": "val_17",
        "project": "packet-sniffer",
        "question": "Which transport protocols can your sniffer identify?",
        "answer": "It inspects the protocol byte in the IPv4 header to identify TCP (6), UDP (17), and ICMP (1).",
        "intended_tier": "3"
    },
    {
        "id": "val_18",
        "project": "packet-sniffer",
        "question": "What networking socket configuration is used?",
        "answer": "AF_INET raw sockets in Python.",
        "intended_tier": "2"
    },
    {
        "id": "val_19",
        "project": "packet-sniffer",
        "question": "How does the sniffer process packets?",
        "answer": "It reads incoming network packets from the socket and prints the protocol information.",
        "intended_tier": "1"
    },
    {
        "id": "val_20",
        "project": "packet-sniffer",
        "question": "How does your packet sniffer stream captured packets into Kafka for real-time SIEM analytics?",
        "answer": "We deployed an Apache Kafka cluster with snappy compression and mirror-maker to stream 50,000 packets/sec into Elasticsearch and Splunk.",
        "intended_tier": "0"
    },
    {
        "id": "val_21",
        "project": "packet-sniffer",
        "question": "How do you calculate the header length from the IHL field in IPv4?",
        "answer": "The Internet Header Length (IHL) is the lower 4 bits of the first byte, representing the length in 32-bit words. Multiplying by 4 yields the exact header size in bytes, typically 20 bytes without options.",
        "intended_tier": "4"
    },

    # --- RAFT-KV (8 items) ---
    {
        "id": "val_22",
        "project": "raft-kv",
        "question": "How did you tune leader election timeouts in your Raft implementation?",
        "answer": "We implemented randomized election timeouts between 150ms and 300ms as prescribed in Section 5.2 of the Raft paper, with 50ms heartbeat intervals to prevent split votes across our 3-node AWS cluster.",
        "intended_tier": "4"
    },
    {
        "id": "val_23",
        "project": "raft-kv",
        "question": "What throughput benchmark did your Raft cluster achieve?",
        "answer": "Tested on 3 AWS c5.xlarge nodes, we reached a maximum stable write throughput of 45,000 req/sec with 99.99% linearizability.",
        "intended_tier": "3"
    },
    {
        "id": "val_24",
        "project": "raft-kv",
        "question": "How did you test your cluster under network partitions?",
        "answer": "We tested network partition recovery via simulated dropped TCP connections during concurrent Jepsen partition tests.",
        "intended_tier": "3"
    },
    {
        "id": "val_25",
        "project": "raft-kv",
        "question": "What consensus algorithm does Raft-KV implement?",
        "answer": "It implements Raft leader election and log replication.",
        "intended_tier": "2"
    },
    {
        "id": "val_26",
        "project": "raft-kv",
        "question": "How did you scale Raft-KV to a 50-node cluster using Paxos consensus?",
        "answer": "We integrated Multi-Paxos and EPaxos engines across 50 nodes to achieve 500,000 transactions per second with Byzantine fault tolerance.",
        "intended_tier": "0"
    },
    {
        "id": "val_27",
        "project": "raft-kv",
        "question": "How does Raft-KV handle log compaction when log files grow indefinitely?",
        "answer": "We periodically snapshot the applied state machine state and truncate committed log entries up to the snapshot index, sending InstallSnapshot RPCs to lagging followers.",
        "intended_tier": "4"
    },
    {
        "id": "val_28",
        "project": "raft-kv",
        "question": "What happens when a partitioned leader reconnects with a lower term number?",
        "answer": "When a partitioned former leader reconnects and observes an AppendEntries RPC or Vote request with a higher term number, it immediately steps down to follower state and updates its term.",
        "intended_tier": "4"
    },
    {
        "id": "val_29",
        "project": "raft-kv",
        "question": "How many nodes were tested in your Raft cluster benchmark?",
        "answer": "Strictly 3 nodes on AWS c5.xlarge instances.",
        "intended_tier": "3"

    },

    # --- WAL-ENGINE (7 items) ---
    {
        "id": "val_30",
        "project": "wal-engine",
        "question": "How does WAL-Engine eliminate OS page cache double-buffering during high-throughput logging?",
        "answer": "We open write-ahead log files with O_DIRECT flag in C++, which bypasses the Linux page cache and writes directly to NVMe storage using 4KB memory-aligned circular buffers.",
        "intended_tier": "4"
    },
    {
        "id": "val_31",
        "project": "wal-engine",
        "question": "What fsync latency and throughput did WAL-Engine achieve?",
        "answer": "We achieved a p99 fsync latency of 1.13ms under 10,000 concurrent commits per second on NVMe storage.",
        "intended_tier": "3"
    },
    {
        "id": "val_32",
        "project": "wal-engine",
        "question": "How do you detect corrupted log records after sudden system crashes?",
        "answer": "Each log record header stores a CRC32-IEEE checksum computed during write. On replay, CRC is recomputed; torn writes are detected and safely truncated.",
        "intended_tier": "4"
    },
    {
        "id": "val_33",
        "project": "wal-engine",
        "question": "How are log files segmented?",
        "answer": "Log segment files are rolled at 64MB boundaries.",
        "intended_tier": "3"
    },
    {
        "id": "val_34",
        "project": "wal-engine",
        "question": "How does the storage engine write data?",
        "answer": "It writes records to a sequential log file on disk.",
        "intended_tier": "1"
    },
    {
        "id": "val_35",
        "project": "wal-engine",
        "question": "Did you test crash recovery with fault injection?",
        "answer": "Yes, we ran 500 fault-injection cycles using sudden kill -9 and simulated power cuts with 0 corrupt bytes detected.",
        "intended_tier": "3"
    },
    {
        "id": "val_36",
        "project": "wal-engine",
        "question": "Why is 4KB alignment necessary when using O_DIRECT in Linux storage engines?",
        "answer": "The Linux kernel requires both the user-space memory buffer address, file offset, and write length to be integer multiples of the underlying block device sector size (typically 4096 bytes); otherwise write() fails with EINVAL.",
        "intended_tier": "4"
    },

    # --- FLOW-LIMITER (7 items) ---
    {
        "id": "val_37",
        "project": "flow-limiter",
        "question": "How does your sliding-window rate limiter prevent boundary burst spikes in Go?",
        "answer": "We implemented a sliding log algorithm evaluated atomically inside Redis using a Lua script. Each incoming request records its Unix microsecond timestamp in a sorted set (ZSET), trims expired entries with ZREMRANGEBYSCORE, and counts remaining members with ZCARD in a single round-trip.",
        "intended_tier": "4"
    },
    {
        "id": "val_38",
        "project": "flow-limiter",
        "question": "What throughput and latency did Flow-Limiter sustain?",
        "answer": "It processes 80,000 req/sec with p99 latency under 0.8ms.",
        "intended_tier": "3"
    },
    {
        "id": "val_39",
        "project": "flow-limiter",
        "question": "What happens if Redis becomes unavailable or drops heartbeat?",
        "answer": "Flow-Limiter activates an in-memory local leaky bucket fallback to maintain rate limiting without dropping traffic.",
        "intended_tier": "3"
    },
    {
        "id": "val_40",
        "project": "flow-limiter",
        "question": "How do you rate limit requests?",
        "answer": "We track client requests using Redis keys and check if they exceed the limit.",
        "intended_tier": "1"
    },
    {
        "id": "val_41",
        "project": "flow-limiter",
        "question": "Why did you use a Redis Lua script instead of standard Redis pipeline commands?",
        "answer": "A Redis pipeline executes multiple round trips and lacks atomicity between read and write. A Lua script executes atomically on the Redis server, guaranteeing no race condition between checking token count and adding the timestamp.",
        "intended_tier": "4"
    },
    {
        "id": "val_42",
        "project": "flow-limiter",
        "question": "What rate limiting algorithm was chosen?",
        "answer": "Sliding log algorithm with Redis sorted sets.",
        "intended_tier": "2"
    },
    {
        "id": "val_43",
        "project": "flow-limiter",
        "question": "Does Flow-Limiter support multi-region Cassandra token bucket syncing?",
        "answer": "We synchronized rate limit buckets globally using Apache Cassandra with lightweight transactions (LWT) across Europe and Asia.",
        "intended_tier": "0"
    },

    # --- FEATURE-FORGE & TABULAR-BOOST (7 items) ---
    {
        "id": "val_44",
        "project": "feature-forge",
        "question": "How does Feature-Forge achieve p95 lookup latencies under 4.2ms for 2M feature keys?",
        "answer": "We ingest streaming events via Kafka and commit micro-batches into DuckDB columnar storage, validating schemas with Pydantic and maintaining point lookups under 4.2ms for 2M keys.",
        "intended_tier": "4"
    },
    {
        "id": "val_45",
        "project": "feature-forge",
        "question": "What daily streaming data volume did you verify in Feature-Forge?",
        "answer": "We verified a daily streaming throughput of 2.5 TB/day in our production cluster.",
        "intended_tier": "3"
    },
    {
        "id": "val_46",
        "project": "feature-forge",
        "question": "Did Feature-Forge process 10 TB/day of streaming data?",
        "answer": "No, our verified maximum production volume was 2.5 TB/day; it was never tested at 10 TB/day.",
        "intended_tier": "3"
    },
    {
        "id": "val_47",
        "project": "feature-forge",
        "question": "What tools power your feature store?",
        "answer": "Kafka and DuckDB with Pydantic schema validation.",
        "intended_tier": "2"
    },
    {
        "id": "val_48",
        "project": "tabular-boost",
        "question": "How did you achieve a 3.4x inference speedup in Tabular-Boost?",
        "answer": "We exported trained LightGBM models to ONNX and used ONNX Runtime with FP32 CPU graph optimizations to fuse redundant operations, increasing inference throughput by 3.4x.",
        "intended_tier": "4"
    },
    {
        "id": "val_49",
        "project": "tabular-boost",
        "question": "What was the AUC-ROC improvement achieved with LightGBM?",
        "answer": "Baseline AUC-ROC improved from 0.81 to 0.88 by tuning feature interaction depth to 6.",
        "intended_tier": "3"
    },
    {
        "id": "val_50",
        "project": "tabular-boost",
        "question": "How did you train your 70B parameter LLM model across 16 NVIDIA A100 GPUs?",
        "answer": "We used PyTorch FSDP (Fully Sharded Data Parallel) and FlashAttention-2 to fine-tune a 70B Llama model on 16x 80GB A100 GPUs.",
        "intended_tier": "0"
    },
]

def main():
    out_dir = Path("data/grader_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_path = out_dir / "labeling_sheet.csv"
    tiers_path = out_dir / "intended_tiers.csv"

    # Write labeling_sheet.csv: id, project, question, answer, human_score (human_score blank, NO tier column)
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "project", "question", "answer", "human_score"])
        for item in ITEMS:
            writer.writerow([item["id"], item["project"], item["question"], item["answer"], ""])

    # Write intended_tiers.csv: id, intended_tier
    with open(tiers_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "intended_tier"])
        for item in ITEMS:
            writer.writerow([item["id"], item["intended_tier"]])

    print(f"Generated {len(ITEMS)} items:")
    print(f" - {sheet_path} (human_score blank, zero tier labels)")
    print(f" - {tiers_path} (intended ground truth preserved separately)")

if __name__ == "__main__":
    main()
