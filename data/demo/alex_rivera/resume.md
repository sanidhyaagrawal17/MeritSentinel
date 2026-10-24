# Alex Rivera
Alex Rivera | San Francisco, CA | alex.rivera@example.com | (555) 234-5678
## Education
B.S. in Computer Science, University of California, Berkeley (CGPA: 3.8 / 4.0, Graduated May 2021)
## Experience
Senior Infrastructure Engineer, CloudScale Systems (June 2021 - Present)
- Designed distributed replication topology across 3 geographic zones.
- Implemented low-latency write ahead logs with disk sync.
## Projects
Raft-KV Distributed Store (Go, Raft, gRPC):
- Implemented Raft consensus protocol with leader election timeouts of 150-300ms maintaining 99.99% linearizability.
- Sustained 45,000 req/sec write throughput under 3-node cluster benchmark.
- Claim: Handled 100,000 req/sec write throughput with zero packet loss under split-brain network partition.
- Claim: Built Paxos consensus engine and migrated infrastructure to Kubernetes with Istio service mesh.
WAL-Engine (C++, Python, POSIX):
- Engineered crash-resilient write-ahead logger with p99 fsync latency of 1.13ms using O_DIRECT on NVMe.
- Achieved zero byte corruption across 500 simulated sudden kernel panic injection cycles.
Flow-Limiter (Go, Redis):
- Distributed sliding-window token bucket limiter processing 80,000 req/sec at sub-millisecond p99 latency.
## Skills
Languages: Go, C++, Python, SQL
Distributed Systems: Raft, gRPC, Redis, POSIX I/O, Consensus protocols
Tools: Docker, Git, Linux, Prometheus