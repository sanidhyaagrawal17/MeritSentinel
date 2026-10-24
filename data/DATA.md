# Data Statement & Fictional Personas

All profiles, resumes, project portfolios, and interview answers contained in `data/demo/` are 100% fictional test fixtures designed solely for automated testing, adversarial evaluation, and demonstrator flows for **ResumeProof**.

## Demo Personas:
1. **Alex Rivera**: Backend & Systems Engineer (5 years exp equivalent). Focus: high-throughput distributed systems, Go/Python, Raft consensus, key-value stores.
2. **Priya Nair**: Data & ML Engineer (3 years exp equivalent). Focus: tabular ML, PyTorch inference pipelines, feature stores, model latency optimization.
3. **Sam Okafor**: Junior B.Tech CSE Student (Entry-level baseline). Focus: REST APIs, SQLite/PostgreSQL caching, networking fundamentals.

## Planted Evaluation Traps:
Each persona portfolio contains:
- 2 supported claims (backed by code, design notes, and concrete metrics).
- 2 partial claims (real project, but inflated numbers or missing specific benchmarks).
- 1-2 unsupported claims (pure hallucination / buzzwords never implemented in the project docs).

## Privacy & Local Isolation Guarantee:
All user data processed by ResumeProof remains strictly on the user's local machine in git-ignored paths. No candidate data is uploaded or transmitted externally.
