# Raft-KV Distributed Key-Value Store
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
