# AGENTS.md

This file gives coding agents, especially Codex-style agents, the full project intent, architecture rules, implementation expectations, and coding conventions for the `Private Local AI Data Standardization and Enrichment Assistant` project.

The goal of this file is to prevent the agent from treating the project as a generic data-cleaning app. This project has a specific architecture and should be implemented according to the design below.

Architecture is already finalized.

Do not redesign the architecture unless explicitly requested.

Do not collapse modules into fewer files.

Prefer extending existing modules instead of creating parallel structures.

Before creating a new file:

1. Check whether an existing module already owns that responsibility.
2. Reuse existing abstractions.
3. Avoid duplicate implementations.
4. Avoid creating helper.py or utils.py unless absolutely necessary.

Keep the MVP simple.

Avoid introducing:

- LangChain
- ChromaDB
- Kafka
- Spark
- Multi-agent systems
- Microservices

unless explicitly requested.

---

## 1. High-Level Instruction for Coding Agents

You are helping build a local-first AI-assisted data standardization and enrichment assistant.

Do not reduce this project to basic data cleaning. The important feature is semantic, rule-based, auditable standardization of messy real-world tabular data.

The project should follow this core workflow:

```text
Upload raw data
-> Profile columns
-> Detect semantic meaning
-> Generate suggested rules using local AI
-> Human reviews rules
-> Apply approved rules deterministically
-> Enrich data using local knowledge base and local LLM fallback
-> Record audit trail
-> Export cleaned data and report
```

The system should be private by default. Do not add cloud AI APIs unless the user explicitly asks.

---

## 2. Project Name

```text
Private Local AI Data Standardization and Enrichment Assistant
```

Short internal name:

```text
ai_data_assistant
```

---

## 3. Main Design Philosophy

### 3.1 AI suggests, rules execute

The local LLM should not directly edit datasets. It should suggest structured transformation rules. The rule engine applies only approved rules.

Correct design:

```text
LLM -> suggested_rules.json -> human review -> approved rules -> RuleEngine -> standardized dataset
```

Incorrect design:

```text
LLM -> directly rewrites the dataset
```

### 3.2 Human review is required

All AI-generated rules should be reviewed by the user.

The user can:

- accept a rule
- edit a rule
- reject a rule

### 3.3 Every transformation must be auditable

Whenever the system changes a value, create an audit record.

An audit record should be able to answer:

- What changed?
- Which row changed?
- Which column changed?
- What was the original value?
- What is the new value?
- Which rule caused the change?
- When did it happen?

### 3.4 Local-first privacy

Default AI provider should be Ollama with a local model.

Do not add OpenAI, Anthropic, Gemini, or other cloud APIs unless explicitly requested by the user.

---

## 4. Primary MVP Domain

The first MVP should focus on job-posting datasets.

Important columns in the MVP:

- job title
- location
- salary
- skills
- description
- company
- experience

Examples:

```text
title: "train model" -> standardized_title: "AI Engineer"
location: "HN" -> city: "Hà Nội"
salary: "20-30 trieu" -> salary_min: 20000000, salary_max: 30000000
skills: "pytorch, ml" -> detected_skills: ["PyTorch", "Machine Learning"]
```

---

## 5. Expected Repository Structure

The user already created the following structure. Respect it. Do not randomly move files unless there is a strong reason.

```text
ai_data_assistant/
├── Makefile
├── pyproject.toml
├── app/
│   ├── components/
│   └── pages/
├── audit/
├── config/
├── core/
│   ├── ai/
│   ├── enrichment/
│   ├── pipeline/
│   ├── profiling/
│   ├── rules/
│   ├── semantic/
│   └── standardization/
├── extensions/
│   └── agents/
├── knowledge/
├── models/
├── services/
├── storage/
└── viz/
```

---

## 6. Layer Responsibilities

### 6.1 `models/`

Contains Pydantic models and core data schemas.

Expected files:

```text
models/dataset.py
models/column_profile.py
models/semantic_tag.py
models/rule.py
models/audit_entry.py
models/pipeline_run.py
```

Use Pydantic models where possible because the app needs structured validation.

Suggested responsibilities:

- `Dataset`: dataset ID, file name, row count, column count, version paths.
- `ColumnProfile`: dtype, null percentage, cardinality, top values, samples.
- `SemanticTag`: column name, semantic type, confidence, evidence.
- `Rule`: rule ID, rule type, target column, output column, conditions, action, confidence, status.
- `AuditEntry`: old value, new value, rule ID, row index, timestamp.
- `PipelineRun`: run ID, dataset ID, stage statuses, artifact paths.

---

### 6.2 `core/profiling/`

Implements dataset and column profiling.

Files:

```text
core/profiling/column_profiler.py
core/profiling/dataset_profiler.py
core/profiling/metadata_builder.py
```

Expected behavior:

- Calculate dtype.
- Calculate null count and null percentage.
- Calculate unique count and cardinality ratio.
- Extract top values.
- Extract sample values.
- Build compact metadata for the semantic and AI layers.

Do not call the LLM from profiling code.

---

### 6.3 `core/semantic/`

Detects semantic meaning of columns.

Files:

```text
core/semantic/base_detector.py
core/semantic/job_detector.py
core/semantic/location_detector.py
core/semantic/salary_detector.py
core/semantic/detector_registry.py
```

Semantic detection should initially be rule-based using:

- column names
- top values
- sample values
- regex patterns
- knowledge base hints

Expected semantic types:

```text
JOB_TITLE
LOCATION
SALARY
SKILL_LIST
COMPANY
DESCRIPTION
EXPERIENCE
UNKNOWN
```

Do not hard-code everything inside one function. Use detector classes and a registry.

---

### 6.4 `core/ai/`

Wraps local LLM support.

Files:

```text
core/ai/base_provider.py
core/ai/ollama_provider.py
core/ai/prompt_builder.py
core/ai/response_parser.py
core/ai/prompts/rule_generation.txt
core/ai/prompts/semantic_detection.txt
core/ai/prompts/enrichment.txt
```

Important rules:

- Default provider is Ollama.
- Default model can be `qwen3:8b` or configurable in `config/default.toml`.
- Prompt builder should pass compact metadata, not full raw data.
- Response parser must validate LLM output.
- Invalid JSON from the LLM should not crash the app. Return a useful error or fallback.

Expected provider interface:

```python
class BaseProvider:
    def generate(self, prompt: str, **kwargs) -> str:
        ...
```

Expected Ollama behavior:

- Call local Ollama HTTP endpoint.
- Handle connection errors.
- Return text response.
- Never silently send data to external APIs.

---

### 6.5 `core/rules/`

Defines rule objects and rule execution.

Files:

```text
core/rules/base_rule.py
core/rules/mapping_rule.py
core/rules/regex_rule.py
core/rules/transform_rule.py
core/rules/validation_rule.py
core/rules/rule_engine.py
core/rules/rule_serializer.py
```

Supported rule types:

- `mapping`: map known variants to canonical value.
- `regex`: extract or normalize values using regex.
- `transform`: apply deterministic transformation.
- `validation`: check whether values satisfy a constraint.

Rule execution should be deterministic and testable.

The rule engine should return:

- transformed dataframe
- list of audit entries
- rule impact summary

Do not mix Streamlit UI code into the rule engine.

---

### 6.6 `core/standardization/`

Contains domain-specific standardizers.

Files:

```text
core/standardization/base_standardizer.py
core/standardization/job_title_standardizer.py
core/standardization/location_standardizer.py
core/standardization/salary_standardizer.py
core/standardization/experience_standardizer.py
core/standardization/standardizer_registry.py
```

Expected examples:

- Job title standardizer normalizes job title variants.
- Location standardizer maps city abbreviations to canonical cities.
- Salary standardizer parses Vietnamese salary formats.
- Experience standardizer parses years of experience.

Examples:

```text
"HN" -> "Hà Nội"
"TP.HCM" -> "Hồ Chí Minh"
"20-30tr" -> salary_min=20000000, salary_max=30000000
"train model" -> "AI Engineer"
```

---

### 6.7 `core/enrichment/`

Adds new information after standardization.

Files:

```text
core/enrichment/base_enricher.py
core/enrichment/kb_enricher.py
core/enrichment/llm_enricher.py
core/enrichment/enricher_registry.py
```

First use the knowledge base. Use LLM only as fallback.

Examples:

```text
standardized_title="AI Engineer" -> domain="AI/ML"
city="Hà Nội" -> region="North Vietnam"
skill="PyTorch" -> skill_group="Deep Learning"
```

---

### 6.8 `knowledge/`

Contains local JSON knowledge base files.

Files:

```text
knowledge/job_titles.json
knowledge/cities.json
knowledge/salary_patterns.json
knowledge/skills.json
knowledge/domains.json
```

These files should be easy to edit manually.

Recommended shape examples:

`job_titles.json`:

```json
{
  "AI Engineer": {
    "aliases": [
      "ai engineer",
      "machine learning engineer",
      "train model",
      "deep learning engineer"
    ],
    "domain": "AI/ML",
    "level_keywords": ["intern", "junior", "senior", "lead"]
  }
}
```

`cities.json`:

```json
{
  "Hà Nội": {
    "aliases": ["ha noi", "hanoi", "hn", "hà nội"],
    "region": "North Vietnam"
  },
  "Hồ Chí Minh": {
    "aliases": ["hcm", "ho chi minh", "tp.hcm", "sai gon", "sài gòn"],
    "region": "South Vietnam"
  }
}
```

---

### 6.9 `audit/`

Handles change history and rule history.

Files:

```text
audit/change_log.py
audit/rule_history.py
```

Expected behavior:

- Save each transformation event.
- Save rule versions.
- Allow report generation later.

Do not make audit optional for standardization changes. If data changes, audit should record it.

---

### 6.10 `services/`

Service layer connects core logic to the UI.

Files:

```text
services/ingest_service.py
services/profiling_service.py
services/semantic_service.py
services/rule_generation_service.py
services/standardization_service.py
services/enrichment_service.py
services/audit_service.py
services/export_service.py
```

Services should be thin orchestration layers.

Good service behavior:

```text
Service receives input -> calls core module -> stores artifact/session -> returns result
```

Avoid putting complex business logic only inside Streamlit pages.

---

### 6.11 `app/`

Streamlit UI.

Pages:

```text
app/pages/01_upload.py
app/pages/02_profile.py
app/pages/03_semantic.py
app/pages/04_rules.py
app/pages/05_standardize.py
app/pages/06_enrich.py
app/pages/07_audit.py
app/pages/08_export.py
```

Expected flow:

1. Upload data.
2. Preview data.
3. Show profile.
4. Show semantic detection.
5. Generate and review rules.
6. Apply standardization.
7. Enrich data.
8. Review audit.
9. Export artifacts.

Components:

```text
app/components/data_preview.py
app/components/profile_card.py
app/components/rule_editor.py
```

Keep UI code separate from core logic.

---

### 6.12 `storage/`

Handles artifacts and sessions.

Files:

```text
storage/artifact_store.py
storage/session_store.py
```

Expected responsibilities:

- Save uploaded raw files.
- Save versioned datasets.
- Save rules.
- Save audit logs.
- Save reports.
- Manage Streamlit session state helpers.

Use version naming:

```text
v0_raw
v1_standardized
v2_enriched
```

---

### 6.13 `viz/`

Contains chart builders.

Files:

```text
viz/profile_charts.py
viz/semantic_charts.py
viz/audit_charts.py
viz/rule_impact_charts.py
```

Do not put chart code directly into services or core modules.

---

### 6.14 `extensions/agents/`

Future-only extension area.

File:

```text
extensions/agents/agent_executor.py
```

Do not implement complex agent behavior in the MVP unless requested.

---

## 7. Implementation Order for Agents

When asked to start coding, follow this order unless the user asks otherwise:

1. `pyproject.toml`
2. `config/default.toml`
3. Pydantic models in `models/`
4. Knowledge base JSON files in `knowledge/`
5. Ingestion service
6. Profiling core and service
7. Semantic detectors and service
8. Rule models and rule engine
9. Standardizers
10. Audit models and service
11. Enrichment core and service
12. AI provider, prompt builder, and response parser
13. Streamlit pages
14. Export service
15. Visualization charts
16. Tests
17. Demo dataset and demo script

Do not start with the LLM. Build deterministic pieces first.

---

## 8. Coding Style

### 8.1 Python style

- Use Python 3.11+.
- Use type hints.
- Prefer small, testable classes and functions.
- Use Pydantic for structured models.
- Use pathlib instead of raw string paths.
- Avoid global mutable state.
- Keep Streamlit session state inside UI/storage helpers.

### 8.2 Error handling

Handle common errors clearly:

- file not found
- unsupported file format
- invalid JSON
- Ollama not running
- missing required columns
- malformed rules
- empty dataset

Do not fail silently.

### 8.3 Dataframe backend

Prefer Polars for backend transformations.

Pandas can be used for:

- Streamlit preview
- small demo data
- compatibility when needed

### 8.4 Config

Do not hard-code model names, paths, or settings deep in modules.

Use:

```text
config/default.toml
config/settings.py
```

Example config keys:

```toml
[ai]
provider = "ollama"
model = "qwen3:8b"
base_url = "http://localhost:11434"

data]
artifact_dir = "artifacts"
max_preview_rows = 100

[rules]
default_confidence_threshold = 0.70
require_human_review = true
```

---

## 9. Rule Schema Expectations

A rule should be structured and serializable.

Suggested fields:

```text
rule_id
name
rule_type
target_column
output_column
semantic_type
conditions
action
confidence
status
explanation
created_at
updated_at
created_by
```

Recommended statuses:

```text
suggested
accepted
edited
rejected
applied
```

Recommended rule types:

```text
mapping
regex
transform
validation
```

Example:

```json
{
  "rule_id": "job_title_ai_engineer_001",
  "name": "Map AI-related titles to AI Engineer",
  "rule_type": "mapping",
  "target_column": "title",
  "output_column": "standardized_title",
  "semantic_type": "JOB_TITLE",
  "conditions": {
    "contains_any": [
      "ai engineer",
      "machine learning",
      "train model",
      "deep learning"
    ]
  },
  "action": {
    "set_value": "AI Engineer"
  },
  "confidence": 0.86,
  "status": "suggested",
  "explanation": "These values describe AI or machine learning engineering roles."
}
```

---

## 10. Audit Schema Expectations

Every applied change should create an audit entry.

Suggested fields:

```text
audit_id
run_id
dataset_id
row_index
column
old_value
new_value
rule_id
rule_type
timestamp
confidence
note
```

Example:

```json
{
  "audit_id": "audit_000001",
  "run_id": "run_001",
  "dataset_id": "dataset_001",
  "row_index": 12,
  "column": "title",
  "old_value": "train model",
  "new_value": "AI Engineer",
  "rule_id": "job_title_ai_engineer_001",
  "rule_type": "mapping",
  "timestamp": "2026-06-09T22:30:00",
  "confidence": 0.86,
  "note": "Applied accepted mapping rule."
}
```

---

## 11. Prompt Design Rules

Prompt files live in:

```text
core/ai/prompts/
```

Prompt files:

```text
semantic_detection.txt
rule_generation.txt
enrichment.txt
```

Prompts should ask the LLM to return JSON only when possible.

Prompt should include:

- project role
- task
- dataset metadata
- semantic tags
- examples of expected JSON
- strict output requirements

The LLM should be told:

- Do not invent columns.
- Use only provided metadata.
- Return confidence score.
- Return explanation.
- Return valid JSON.
- Prefer safe rules.
- Mark uncertain rules with lower confidence.

---

## 12. Testing Expectations

Add tests when implementing core logic.

Important tests:

- salary parsing
- location normalization
- job title mapping
- rule serialization
- rule engine application
- audit entry creation
- semantic detection
- response parser JSON validation

Example test names:

```text
tests/test_salary_standardizer.py
tests/test_location_standardizer.py
tests/test_rule_engine.py
tests/test_semantic_detectors.py
tests/test_response_parser.py
```

---

## 13. Demo Dataset Recommendation

Create a small demo dataset for development.

Suggested columns:

```text
title
company
location
salary
skills
description
experience
```

Suggested messy values:

```text
AI Engineer
A.I Engineer
train model
Machine Learning Engineer
Data analyst
BI analyst
HN
Ha Noi
Hà Nội
HCM
TP.HCM
20-30tr
upto 25M
15 triệu - 20 triệu
Python, ML
pytorch, deep learning
SQL, Power BI
```

This dataset should demonstrate why the project is useful.

---

## 14. What Not to Do

Do not:

- Build only a missing-value/duplicate cleaner.
- Let the LLM directly rewrite the dataset.
- Send private data to cloud APIs by default.
- Put all logic inside Streamlit pages.
- Skip audit logging.
- Hard-code all rules in one giant function.
- Ignore the existing architecture.
- Build future agent/RAG features before the MVP works.

---

## 15. Definition of Done for MVP

The MVP is done when the app can:

1. Upload a CSV job dataset.
2. Display raw preview.
3. Profile columns.
4. Detect semantic tags.
5. Generate suggested standardization rules.
6. Let user accept/edit/reject rules.
7. Apply accepted rules.
8. Create standardized columns.
9. Enrich data using local JSON knowledge base.
10. Save audit log.
11. Show before/after comparison.
12. Export cleaned data and report.

---

## 16. Suggested First Coding Task

If the user asks to begin coding, start with foundational files:

```text
pyproject.toml
config/default.toml
config/settings.py
models/column_profile.py
models/semantic_tag.py
models/rule.py
models/audit_entry.py
models/dataset.py
models/pipeline_run.py
knowledge/job_titles.json
knowledge/cities.json
knowledge/salary_patterns.json
knowledge/skills.json
knowledge/domains.json
```

Then implement profiling and semantic detection before LLM integration.

---

## 17. Graduation Defense Angle

This project should be explainable as a serious applied AI/data engineering system.

Key talking points:

- Private local AI assistant.
- Human-in-the-loop rule review.
- Deterministic rule engine.
- Semantic column understanding.
- Knowledge-driven enrichment.
- Full audit trail.
- Practical use case with messy job datasets.
- More valuable than simple cleaning because it standardizes business meaning.

---

## 18. One-Sentence Summary for Agents

Build a local, private, human-reviewed, rule-based AI assistant that turns messy tabular data into standardized, enriched, and auditable datasets, starting with job-posting data as the MVP domain.
