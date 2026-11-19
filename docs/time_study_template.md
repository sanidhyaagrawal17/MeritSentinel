# Phase 5 Human Time-Study Protocol & Evaluation Template

## 1. Study Objective
To empirically quantify the efficiency gains, accuracy improvements, and cognitive load reduction achieved by using **MeritSentinel** compared to traditional manual resume tailoring and verification against target job descriptions.

---

## 2. Experimental Design
- **Subjects**: Evaluators / Human Career Advisors (N ≥ 5 target evaluators).
- **Task**: Tailor and verify a baseline engineering candidate profile against a production job description (JD) containing both supported qualifications and out-of-domain requirements.
- **Conditions**:
  1. **Condition A (Manual Baseline)**: The evaluator manually cross-references the candidate's master claim repository, identifies supported skills vs missing gaps, rewrites resume bullet points to emphasize relevant claims, checks metrics for truthfulness, and typesets a 1-page PDF.
  2. **Condition B (MeritSentinel Assisted)**: The evaluator uses MeritSentinel's automated pipeline (Hybrid Matcher for JD alignment, Strict Verifier for AST token/metric checking, and Single-Page ATS PDF Exporter).

---

## 3. Metrics Log Sheet (Per Participant)

| Evaluation Participant ID: | `P-001` (Example) |
| :--- | :--- |
| **Date & Time:** | `YYYY-MM-DD HH:MM` |
| **Target Job Description:** | Senior Backend / Distributed Systems Engineer |
| **Baseline Profile:** | Jordan Hayes (PCIT, 4 Verified Projects) |

### 3.1 Task Duration Tracking (Minutes)

| Phase / Task Component | Manual Baseline (Condition A) | MeritSentinel (Condition B) | Absolute Time Saved | % Time Saved |
| :--- | :---: | :---: | :---: | :---: |
| **Task 1: Requirement Gap Analysis**<br>*(Scanning JD, comparing to claims, identifying missing tools)* | `___._ min` | `___._ min` | `___._ min` | `___._ %` |
| **Task 2: Claim Verification & Fact-Checking**<br>*(Checking metrics, tool names, dates, unbacked buzzwords)* | `___._ min` | `___._ min` | `___._ min` | `___._ %` |
| **Task 3: Resume Tailoring & Drafting**<br>*(Drafting 15-20 compliant bullet points matching JD)* | `___._ min` | `___._ min` | `___._ min` | `___._ %` |
| **Task 4: Compilation & Formatting**<br>*(Aligning margins, page budget enforcement, ATS compliance)* | `___._ min` | `___._ min` | `___._ min` | `___._ %` |
| **TOTAL TIME ELAPSED** | **`___._ min`** | **`___._ min`** | **`___._ min`** | **`___._ %`** |

---

### 3.2 Output Quality & Verification Audit

| Metric | Manual Baseline | MeritSentinel Automated | Ground Truth Delta |
| :--- | :---: | :---: | :---: |
| **Total Requirements Evaluated** | `10` | `10` | 0 |
| **True Gaps Identified (e.g. Redis, K8s)** | `___ / 2` | `2 / 2` | Automated quarantine |
| **Hallucinated / Unbacked Metrics Injected** | `___` (Human error) | `0` (AST Blocked) | Strict AST gate |
| **Unbacked Tools Leaked into Output** | `___` | `0` | Open-world AST filter |
| **Page Budget Exceeded (> 1 Page)?** | `[ ] Yes  [ ] No` | `[X] No (1 Page Strict)`| ReportLab budget check |

---

### 3.3 Subjective Cognitive Load Rating (NASA-TLX Adapted, 1-10 Scale)
*1 = Very Low / Minimal Effort, 10 = Very High / Exhausting*

| Dimension | Manual Baseline (1-10) | MeritSentinel (1-10) | Reduction (Δ) |
| :--- | :---: | :---: | :---: |
| **Mental Demand** (How much mental activity was required?) | `__ / 10` | `__ / 10` | `__` |
| **Time Pressure** (How rushed or hurried was the pace?) | `__ / 10` | `__ / 10` | `__` |
| **Effort** (How hard did you have to work to maintain accuracy?) | `__ / 10` | `__ / 10` | `__` |
| **Frustration Level** (How insecure, discouraged, or annoyed were you?) | `__ / 10` | `__ / 10` | `__` |
| **Overall Confidence in Truthfulness** (1=Uncertain, 10=Completely Confident) | `__ / 10` | `__ / 10` | `+__` |

---

## 4. Submission & Integration
Completed study sheets should be digitized into JSON format:
```json
[
  {
    "participant_id": "P-001",
    "manual_time_minutes": 42.5,
    "meritsentinel_time_minutes": 3.8,
    "time_saved_percent": 91.06,
    "manual_unbacked_leaks": 2,
    "meritsentinel_unbacked_leaks": 0,
    "manual_cognitive_load": 7.8,
    "meritsentinel_cognitive_load": 2.1
  }
]
```
These will be processed by `scripts/eval_time_study.py` once collected.
