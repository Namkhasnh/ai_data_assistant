# Private Local AI Data Standardization and Enrichment Assistant

> A privacy-first, local-first data standardization and enrichment assistant for turning messy CSV/JSON/Parquet datasets into standardized, enriched, auditable outputs.

This project is intended as a graduation/capstone project. The main goal is not just basic data cleaning such as removing duplicates or filling missing values. The main goal is to build a practical assistant that can understand messy real-world tabular data, detect semantic meanings of columns, suggest standardization rules, let humans review those rules, apply them safely, enrich the data using a local knowledge base and local LLM fallback, and produce traceable outputs.

---

## 1. Project Vision

Real datasets often contain inconsistent values. For example, job datasets crawled from websites may contain job titles like:

- `AI Engineer`
- `A.I Engineer`
- `Machine Learning Engineer`
- `train model`
- `Computer Vision Intern`
- `Data Scientist / AI Engineer`

A normal cleaning script can handle missing values and duplicates, but it usually does not understand that these values may need to be standardized into canonical groups such as:

- `AI Engineer`
- `Data Scientist`
- `Data Analyst`
- `Data Engineer`
- `Software Engineer`

This project tries to solve that type of problem.

The assistant should accept a raw dataset, profile it, understand the semantic meaning of each column, propose transformation rules, let the user review those rules, apply the approved rules, enrich the result with a knowledge base, and export both the cleaned data and an audit report.

---

## 2. Core Principles

### 2.1 Local-first and private

The project should run locally on the user's machine. Sensitive data should not be sent to cloud APIs by default.

The AI provider should use a local LLM through Ollama, such as Qwen3 8B, instead of external services.

### 2.2 Human-in-the-loop

AI should not directly modify data without review. The AI can suggest rules, but the user should be able to accept, edit, or reject every rule.

### 2.3 Rule-based execution after AI suggestion

The LLM is used mainly to suggest rules. The final transformation should be deterministic and explainable through a rule engine.

This means:

1. AI proposes a rule.
2. The user reviews the rule.
3. The rule is saved as structured JSON.
4. The rule engine applies it.
5. The audit layer records what changed.

### 2.4 Traceability

Every change should be explainable:

- Which rule changed the value?
- Which rows were affected?
- What was the old value?
- What is the new value?
- When did the change happen?
- What confidence level did the rule have?

### 2.5 Extensible architecture

The current target is job-related datasets, but the architecture should support future domains such as finance, e-commerce, education, HR, and customer data.

---

## 3. Target Users

The target users are:

- Data analysts who need to clean messy datasets quickly.
- Students building data projects.
- Small teams that want a private data preprocessing tool.
- Companies that cannot upload sensitive datasets to cloud AI tools.

---

## 4. Example Use Case

### Input dataset

A raw job dataset crawled from websites such as TopCV, LinkedIn, or VietnamWorks.

Example columns:

| title        | company  | location    | salary      | skills        | description                  |
| ------------ | -------- | ----------- | ----------- | ------------- | ---------------------------- |
| AI Engineer  | ABC Corp | Ha Noi      | 20-30 trieu | Python, ML    | build model                  |
| train model  | XYZ Ltd  | HN          | upto 25M    | pytorch       | training deep learning model |
| Data analyst | DEF      | Ho Chi Minh | 15M-20M     | SQL, Power BI | dashboard                    |

### Expected output

The assistant should create standardized and enriched columns such as:

| title        | standardized_title | city        | salary_min | salary_max | domain | detected_skills          |
| ------------ | ------------------ | ----------- | ---------- | ---------- | ------ | ------------------------ |
| AI Engineer  | AI Engineer        | Hà Nội      | 20000000   | 30000000   | AI/ML  | Python, Machine Learning |
| train model  | AI Engineer        | Hà Nội      | null       | 25000000   | AI/ML  | PyTorch, Deep Learning   |
| Data analyst | Data Analyst       | Hồ Chí Minh | 15000000   | 20000000   | Data   | SQL, Power BI            |

The system should also produce:

- `cleaned_data.csv`
- `cleaned_data.parquet`
- `rules.json`
- `suggested_rules.json`
- `audit_log.json`
- `cleaning_report.html`
- optional generated Python transformation code

---

## 5. Architecture Overview

The architecture follows a layered pipeline:

```text
Data Ingestion Layer
  CSV / JSON / Parquet -> LazyFrame
        |
        v
Data Profiling Layer
  dtype, null%, cardinality, top values
        |
        v
Semantic Understanding Layer
  "title" / "position" -> JOB_TITLE
        |
        v
AI Rule Generation Layer
  DatasetMetadata -> suggested rules.json
        |
        v
Human Review Layer
  Accept / Edit / Reject per rule
        |
        v
Rule-Based Standardization Engine
  Mapping / Regex / Transform / Validation rules
        |
        v
Knowledge-Driven Enrichment Layer
  KB lookup -> LLM fallback
        |
        v
Audit & Traceability Layer
  rule_id, affected_rows, timestamp
        |
        v
Reporting & Analytics Layer
  before/after diff, rule impact charts
        |
        v
Data Export Layer
  CSV / Parquet / JSON / reports
```

Supporting layers:

- AI supporting layer: Ollama provider, prompt builder, response parser, future agent executor.
- Knowledge base: job titles, cities, salary patterns, skills, domains.
- Dataset versioning: `v0_raw`, `v1_standardized`, `v2_enriched`.

---

## 6. Main Pipeline Stages

### 6.1 Data Ingestion Layer

Responsible for loading user files into a consistent internal format.

Supported input formats:

- CSV
- JSON
- Parquet
- Excel may be added later

Recommended internal backend:

- Polars `LazyFrame` for scalable data processing.
- Pandas can be used for easier Streamlit preview if needed.

Main responsibilities:

- Read raw files.
- Validate file format.
- Store raw dataset as `v0_raw`.
- Create a dataset object with metadata.

Related files:

```text
services/ingest_service.py
models/dataset.py
storage/artifact_store.py
storage/session_store.py
```

---

### 6.2 Data Profiling Layer

Responsible for understanding the technical profile of each column.

Profile information should include:

- Column name
- Data type
- Null count
- Null percentage
- Number of unique values
- Cardinality ratio
- Top values
- Sample values
- Min/max for numeric columns
- String length statistics for text columns

Related files:

```text
core/profiling/dataset_profiler.py
core/profiling/column_profiler.py
core/profiling/metadata_builder.py
models/column_profile.py
services/profiling_service.py
viz/profile_charts.py
```

Expected output model:

```python
ColumnProfile(
    name="title",
    dtype="str",
    null_count=0,
    null_pct=0.0,
    unique_count=120,
    cardinality_ratio=0.24,
    top_values=[...],
    samples=[...]
)
```

---

### 6.3 Semantic Understanding Layer

Responsible for detecting what each column means, not just its data type.

Example:

| Column name    | Detected semantic type |
| -------------- | ---------------------- |
| `title`        | `JOB_TITLE`            |
| `position`     | `JOB_TITLE`            |
| `address`      | `LOCATION`             |
| `salary_range` | `SALARY`               |
| `tech_stack`   | `SKILL_LIST`           |

The first version should use generic, knowledge-driven rule-based detectors. Later versions can add local LLM or embedding detectors behind the same detector interface.

Related files:

```text
core/semantic/base_detector.py
core/semantic/keyword_detector.py
core/semantic/regex_detector.py
core/semantic/detector_registry.py
knowledge/semantic_rules.json
models/semantic_tag.py
services/semantic_service.py
viz/semantic_charts.py
```

Recommended semantic tags:

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

---

### 6.4 AI Rule Generation Layer

Responsible for asking the local LLM to suggest standardization rules based on dataset metadata.

The LLM should receive metadata, not the full dataset unless necessary.

Input:

- Column profiles
- Semantic tags
- Top values
- Sample values
- Existing knowledge base entries

Output:

- Suggested rules in structured JSON format

Related files:

```text
core/ai/base_provider.py
core/ai/ollama_provider.py
core/ai/prompt_builder.py
core/ai/response_parser.py
core/ai/prompts/rule_generation.txt
core/ai/prompts/semantic_detection.txt
core/ai/prompts/enrichment.txt
services/rule_generation_service.py
```

Expected LLM output example:

```json
[
  {
    "rule_id": "job_title_ai_engineer_001",
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
    "explanation": "These values describe AI or machine learning engineering roles."
  }
]
```

Important: The LLM should not directly modify the dataset. It should only propose rules.

---

### 6.5 Human Review Layer

Responsible for allowing the user to review AI-suggested rules.

User actions:

- Accept rule
- Edit rule
- Reject rule

Each rule should show:

- Rule type
- Target column
- Matching pattern
- Output value
- Confidence score
- Explanation
- Preview of affected rows

Confidence badge idea:

- Green: high confidence
- Amber: medium confidence
- Red: low confidence

Related files:

```text
app/pages/04_rules.py
app/components/rule_editor.py
models/rule.py
services/rule_generation_service.py
```

---

### 6.6 Rule-Based Standardization Engine

Responsible for applying approved rules deterministically.

Supported rule types:

- Mapping rule
- Regex rule
- Transform rule
- Validation rule

Related files:

```text
core/rules/base_rule.py
core/rules/mapping_rule.py
core/rules/regex_rule.py
core/rules/transform_rule.py
core/rules/validation_rule.py
core/rules/rule_engine.py
core/rules/rule_serializer.py
core/standardization/base_standardizer.py
core/standardization/job_title_standardizer.py
core/standardization/location_standardizer.py
core/standardization/salary_standardizer.py
core/standardization/experience_standardizer.py
core/standardization/standardizer_registry.py
services/standardization_service.py
```

Standardization examples:

- `HN`, `Ha Noi`, `Hà Nội` -> `Hà Nội`
- `HCM`, `Ho Chi Minh`, `TP.HCM` -> `Hồ Chí Minh`
- `20-30 triệu`, `20M-30M`, `20tr - 30tr` -> `salary_min=20000000`, `salary_max=30000000`
- `train model`, `Machine Learning Engineer` -> `AI Engineer`

---

### 6.7 Knowledge-Driven Enrichment Layer

Responsible for adding useful derived information after standardization.

The enrichment layer should first use local JSON knowledge base files. If no rule or lookup is found, it may use the local LLM as fallback.

Related files:

```text
core/enrichment/base_enricher.py
core/enrichment/kb_enricher.py
core/enrichment/llm_enricher.py
core/enrichment/enricher_registry.py
services/enrichment_service.py
knowledge/job_titles.json
knowledge/cities.json
knowledge/salary_patterns.json
knowledge/skills.json
knowledge/domains.json
```

Enrichment examples:

| Standardized field | Enriched field                             |
| ------------------ | ------------------------------------------ |
| `AI Engineer`      | `domain = AI/ML`                           |
| `Data Analyst`     | `domain = Data Analytics`                  |
| `Python, PyTorch`  | `skill_group = Programming, Deep Learning` |
| `Hà Nội`           | `region = North Vietnam`                   |

---

### 6.8 Audit & Traceability Layer

Responsible for recording every change made by the system.

Each audit entry should contain:

- Dataset ID
- Pipeline run ID
- Row index or row ID
- Column name
- Old value
- New value
- Rule ID
- Rule type
- Timestamp
- Confidence
- User decision if relevant

Related files:

```text
models/audit_entry.py
audit/change_log.py
audit/rule_history.py
services/audit_service.py
viz/audit_charts.py
viz/rule_impact_charts.py
```

Example audit entry:

```json
{
  "run_id": "run_2026_06_09_001",
  "row_index": 12,
  "column": "title",
  "old_value": "train model",
  "new_value": "AI Engineer",
  "rule_id": "job_title_ai_engineer_001",
  "rule_type": "mapping",
  "timestamp": "2026-06-09T22:30:00",
  "confidence": 0.86
}
```

---

### 6.9 Reporting & Analytics Layer

Responsible for showing what happened before and after cleaning.

Report should include:

- Dataset summary
- Column profiles
- Detected semantic types
- Accepted/rejected rules
- Number of affected rows per rule
- Before/after value comparison
- Missing value summary
- Standardized value distribution
- Warnings and low-confidence cases

Related files:

```text
viz/profile_charts.py
viz/audit_charts.py
viz/semantic_charts.py
viz/rule_impact_charts.py
services/export_service.py
app/pages/07_audit.py
app/pages/08_export.py
```

---

### 6.10 Data Export Layer

Responsible for exporting final artifacts.

Expected export files:

```text
cleaned_data.csv
cleaned_data.parquet
cleaned_data.json
rules.json
suggested_rules.json
audit_log.json
cleaning_report.html
cleaning_report.md
```

Related files:

```text
services/export_service.py
storage/artifact_store.py
app/pages/08_export.py
```

---

## 7. Current File Structure

```text
ai_data_assistant/
├── Makefile
├── pyproject.toml
├── app/
│   ├── components/
│   │   ├── data_preview.py
│   │   ├── profile_card.py
│   │   └── rule_editor.py
│   └── pages/
│       ├── 01_upload.py
│       ├── 02_profile.py
│       ├── 03_semantic.py
│       ├── 04_rules.py
│       ├── 05_standardize.py
│       ├── 06_enrich.py
│       ├── 07_audit.py
│       └── 08_export.py
├── audit/
│   ├── change_log.py
│   └── rule_history.py
├── config/
│   ├── default.toml
│   ├── logging_config.py
│   └── settings.py
├── core/
│   ├── ai/
│   │   ├── base_provider.py
│   │   ├── ollama_provider.py
│   │   ├── prompt_builder.py
│   │   ├── response_parser.py
│   │   └── prompts/
│   │       ├── enrichment.txt
│   │       ├── rule_generation.txt
│   │       └── semantic_detection.txt
│   ├── enrichment/
│   │   ├── base_enricher.py
│   │   ├── enricher_registry.py
│   │   ├── kb_enricher.py
│   │   └── llm_enricher.py
│   ├── pipeline/
│   │   ├── context.py
│   │   ├── pipeline.py
│   │   └── step.py
│   ├── profiling/
│   │   ├── column_profiler.py
│   │   ├── dataset_profiler.py
│   │   └── metadata_builder.py
│   ├── rules/
│   │   ├── base_rule.py
│   │   ├── mapping_rule.py
│   │   ├── regex_rule.py
│   │   ├── rule_engine.py
│   │   ├── rule_serializer.py
│   │   ├── transform_rule.py
│   │   └── validation_rule.py
│   ├── semantic/
│   │   ├── base_detector.py
│   │   ├── detector_registry.py
│   │   ├── keyword_detector.py
│   │   └── regex_detector.py
│   └── standardization/
│       ├── base_standardizer.py
│       ├── experience_standardizer.py
│       ├── job_title_standardizer.py
│       ├── location_standardizer.py
│       ├── salary_standardizer.py
│       └── standardizer_registry.py
├── extensions/
│   └── agents/
│       └── agent_executor.py
├── knowledge/
│   ├── cities.json
│   ├── domains.json
│   ├── job_titles.json
│   ├── salary_patterns.json
│   ├── semantic_rules.json
│   └── skills.json
├── models/
│   ├── audit_entry.py
│   ├── column_profile.py
│   ├── dataset.py
│   ├── pipeline_run.py
│   ├── rule.py
│   └── semantic_tag.py
├── services/
│   ├── audit_service.py
│   ├── enrichment_service.py
│   ├── export_service.py
│   ├── ingest_service.py
│   ├── profiling_service.py
│   ├── rule_generation_service.py
│   ├── semantic_service.py
│   └── standardization_service.py
├── storage/
│   ├── artifact_store.py
│   └── session_store.py
└── viz/
    ├── audit_charts.py
    ├── profile_charts.py
    ├── rule_impact_charts.py
    └── semantic_charts.py
```

---

## 8. Recommended Technology Stack

### Core language

- Python 3.11+

### Data processing

- Polars for fast CSV/Parquet processing
- Pandas only when convenient for Streamlit display

### App UI

- Streamlit

### Models and validation

- Pydantic for structured data models

### Local AI

- Ollama
- Qwen3 8B or another local model

### Visualization

- Plotly or Altair for interactive charts

### Config

- TOML config files

### Testing

- pytest

---

## 9. Suggested Dependencies

The final dependency list may change, but the initial version can use:

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "streamlit",
    "polars",
    "pandas",
    "pydantic",
    "toml",
    "plotly",
    "requests",
    "python-dotenv",
    "pytest"
]
```

Optional later dependencies:

```text
openpyxl
pyarrow
rapidfuzz
scikit-learn
langchain
chromadb
```

---

## 10. Suggested MVP Scope

The first working version should focus on one strong use case instead of trying to support every possible dataset.

Recommended MVP domain:

> Job posting datasets

MVP features:

1. Upload CSV.
2. Preview raw data.
3. Profile columns.
4. Detect semantic types for job title, location, salary, and skills.
5. Generate suggested rules using metadata and local LLM.
6. Let user accept/edit/reject rules.
7. Apply approved rules.
8. Enrich standardized data using JSON knowledge base.
9. Show audit logs and before/after comparison.
10. Export cleaned data and report.

---

## 11. Non-goals for the First Version

Do not prioritize these in the first version:

- Full AutoML
- Advanced RAG system
- Cloud deployment
- Multi-user authentication
- Enterprise permission management
- Agent automation
- Support for every domain
- Perfect cleaning accuracy

These can be future extensions.

---

## 12. Future Extensions

Possible future improvements:

- RAG over internal cleaning documentation.
- Agent-based cleaning workflow.
- Auto-generated Python cleaning scripts.
- Support for Excel files.
- Support for multiple domains.
- Domain-specific rule packs.
- Data quality scoring.
- More advanced semantic detection using embeddings.
- Local vector database for knowledge lookup.
- Project templates for common datasets.

---

## 13. How to Think About the System

The system should not be designed as a chatbot that directly edits data.

A better mental model is:

```text
AI suggests -> Human reviews -> Rule engine applies -> Audit records -> Report explains
```

This design makes the project safer, more explainable, and more suitable for a graduation thesis.

---

## 14. Expected Development Order

Recommended order for implementation:

1. Models
2. Knowledge base JSON files
3. Ingestion service
4. Profiling service
5. Semantic detectors
6. Rule models and rule engine
7. Standardizers
8. Audit service
9. Enrichment service
10. AI provider and prompt builder
11. Streamlit UI pages
12. Export service
13. Reporting charts
14. Tests and demo dataset

---

## 15. Success Criteria

The project is successful if a user can:

1. Upload a messy job dataset.
2. See meaningful profiling information.
3. See detected semantic types.
4. Get reasonable suggested standardization rules.
5. Review and approve rules.
6. Apply rules to produce standardized columns.
7. Enrich the dataset with domains, skills, or regions.
8. Inspect an audit log of all changes.
9. Export cleaned data and a report.
10. Explain the whole process in a graduation defense.

---

## 16. Example Demo Story for Graduation Defense

The demo can follow this story:

1. A raw job dataset is crawled from job websites.
2. The dataset has inconsistent job titles, locations, salaries, and skills.
3. The assistant profiles the dataset and identifies important columns.
4. The assistant detects that `title` is a job-title column, `location` is a location column, and `salary` is a salary column.
5. The local LLM suggests standardization rules.
6. The user reviews the rules and approves them.
7. The rule engine applies transformations.
8. The enrichment layer adds domain and skill information.
9. The audit page shows exactly what changed.
10. The export page downloads final data and reports.

This clearly shows the value of the project beyond simple missing-value cleaning.
