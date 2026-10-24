# Flow-Limiter
Distributed sliding-window rate limiter in Go with Redis Lua scripts.

## Performance
- Processes 80,000 req/sec with p99 latency < 0.8ms.
- Atomic Lua script evaluates token count and timestamps in single round-trip.
