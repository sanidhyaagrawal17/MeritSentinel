"""Tests detection of 20 tools outside KNOWN_TOOLS to verify removal of closed-world assumption.
Also reserves a held-out slot for candidate sentences written by the user.
"""

import pytest
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.verifier.strict_verifier import StrictVerifier, KNOWN_TOOLS

# Ground-truth verified claims and facts for demo candidate (Sanidhya baseline)
DEMO_CLAIMS = [
    Claim(id="c1", user_id="u", text="Built RESTful API microservices using FastAPI and SQLite.", tools=["FastAPI", "SQLite", "Python"]),
    Claim(id="c2", user_id="u", text="Trained defect inspection pipeline reaching 61.7% mAP50 on NEU-DET benchmark across 70/20/10 split.", tools=["PyTorch", "YOLO11", "Python"]),
    Claim(id="c3", user_id="u", text="Exported baseline PyTorch model to ONNX runtime format.", tools=["PyTorch", "ONNX", "Python"]),
    Claim(id="c4", user_id="u", text="Built academic research tool indexing arXiv papers in ChromaDB with HNSW vector indexing.", tools=["ChromaDB", "Python", "FastAPI"]),
]

DEMO_FACTS = [
    ProfileFact(id="f1", user_id="u", fact_type=ProfileFactType.NAME, value="Sanidhya Agrawal", is_confirmed=True),
    ProfileFact(id="f2", user_id="u", fact_type=ProfileFactType.DEGREE, value="Bachelor of Technology in Computer Science and Engineering", is_confirmed=True),
    ProfileFact(id="f3", user_id="u", fact_type=ProfileFactType.EDUCATION_INSTITUTION, value="Medi-Caps University", is_confirmed=True),
]

# 20 tools outside KNOWN_TOOLS to verify open-world detection
TWENTY_OUTSIDE_TOOLS = [
    ("Terraform", "Provisioned AWS infrastructure using Terraform modules for reproducible deployment."),
    ("Ansible", "Automated server configuration management using Ansible playbooks across fleet nodes."),
    ("Celery", "Dispatched asynchronous background tasks to Celery workers backed by message brokers."),
    ("Airflow", "Orchestrated DAG workflows for data processing schedules with Apache Airflow."),
    ("dbt", "Engineered data transformation models and staging tables using dbt in analytical pipelines."),
    ("Flink", "Processed real-time streaming event telemetry with Apache Flink stateful operators."),
    ("Cassandra", "Persisted high-throughput time-series event metrics across an Apache Cassandra ring."),
    ("Neo4j", "Modeled knowledge graphs and queried entity relationships using Neo4j Cypher."),
    ("RabbitMQ", "Decoupled backend microservices with message queuing using RabbitMQ exchanges."),
    ("Prometheus", "Scraped real-time API latency metrics and request counters using Prometheus."),
    ("Grafana", "Constructed live monitoring dashboards for cluster telemetry using Grafana."),
    ("Consul", "Managed service discovery and distributed health checking using HashiCorp Consul."),
    ("Vault", "Secured secret storage and automated dynamic credential rotation using HashiCorp Vault."),
    ("Elasticsearch", "Indexed application logs and built full-text search filters with Elasticsearch."),
    ("OpenSearch", "Executed distributed search queries across structured documents in OpenSearch."),
    ("KEDA", "Scaled Kubernetes worker pods based on queue depth metrics using KEDA auto-scaling."),
    ("Spinnaker", "Automated multi-region continuous delivery canary pipelines using Spinnaker."),
    ("ArgoCD", "Managed GitOps declarative application delivery to Kubernetes clusters using ArgoCD."),
    ("Supabase", "Configured real-time authentication and serverless Postgres storage using Supabase."),
    ("ClickHouse", "Engineered columnar analytical aggregation queries on ClickHouse database clusters."),
]

def test_twenty_outside_tools_are_strictly_outside_known_tools():
    """Confirms that NONE of the 20 test tools exist in KNOWN_TOOLS."""
    for tool_name, _ in TWENTY_OUTSIDE_TOOLS:
        assert tool_name not in KNOWN_TOOLS, f"Tool '{tool_name}' must NOT be in KNOWN_TOOLS (violates open-world test)"

def test_open_world_detection_on_twenty_outside_tools():
    """Evaluates detection rate of unbacked tools outside KNOWN_TOOLS.
    Invariant: Every unbacked tool must be flagged/blocked, achieving 100% detection rate."""
    verifier = StrictVerifier()
    
    detected_count = 0
    results = []

    for tool_name, sentence in TWENTY_OUTSIDE_TOOLS:
        res = verifier.verify_sentence(
            sentence=sentence,
            eligible_claims=DEMO_CLAIMS,
            confirmed_facts=DEMO_FACTS,
        )
        is_detected = (res.status == "BLOCKED") and any(
            tool_name.lower() in r.lower() or "unbacked" in r.lower() for r in res.blocking_reasons
        )
        if is_detected:
            detected_count += 1
        results.append({
            "tool": tool_name,
            "sentence": sentence,
            "status": res.status,
            "detected": is_detected,
            "reasons": res.blocking_reasons,
        })

    detection_rate = detected_count / len(TWENTY_OUTSIDE_TOOLS)
    print(f"\nOpen-World Tool Detection: {detected_count} / {len(TWENTY_OUTSIDE_TOOLS)} ({detection_rate * 100:.1f}%)")
    for r in results:
        status_sym = "[BLOCKED]" if r["detected"] else "[PASSED - LEAK]"
        print(f"  {status_sym} Tool '{r['tool']}': {r['reasons']}")

    assert detected_count == len(TWENTY_OUTSIDE_TOOLS), f"Open-world detection failed on {len(TWENTY_OUTSIDE_TOOLS) - detected_count} tools"
    assert detection_rate == 1.0, "Detection rate on tools outside KNOWN_TOOLS must be 100%"

# Reserved held-out slot for user-supplied sentences
USER_HELD_OUT_SENTENCES: list[dict] = [
    # The user can populate their own 15 sentences here
    # Example format:
    # {"text": "Engineered custom distributed consensus with Paxos.", "expected_status": "BLOCKED"}
]

def test_user_held_out_sentence_slot():
    """Executes verification on user-supplied held-out sentences when provided."""
    if not USER_HELD_OUT_SENTENCES:
        pytest.skip("User held-out slot reserved. No custom sentences populated yet.")
    
    verifier = StrictVerifier()
    for item in USER_HELD_OUT_SENTENCES:
        res = verifier.verify_sentence(
            sentence=item["text"],
            eligible_claims=DEMO_CLAIMS,
            confirmed_facts=DEMO_FACTS,
        )
        assert res.status == item["expected_status"], f"Held-out sentence '{item['text']}' failed: got {res.status}, expected {item['expected_status']}"
