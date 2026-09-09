# Graph Report - hiver  (2026-09-09)

## Corpus Check
- Corpus is ~4,599 words - fits in a single context window. You may not need a graph.

## Summary
- 123 nodes · 190 edges · 9 communities
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Intent Taxonomy & Escalation Rules
- Banking77 Preprocessing Benchmark
- Core Agent & Intent Classifier
- Data Inspection & Brand Selection
- Brand Extraction Pipeline
- Topic Profiling & Clustering
- Evaluation Harness
- Golden Set Labeling

## God Nodes (most connected - your core abstractions)
1. `normalize_for_classification()` - 14 edges
2. `Hiver SDE Intern Assignment` - 10 edges
3. `Brand Support Agent Project` - 9 edges
4. `AppleSupport Intent Taxonomy (11 labels)` - 9 edges
5. `classify_intent()` - 7 edges
6. `SupportAgent` - 7 edges
7. `evaluate_variant()` - 7 edges
8. `collect_brand_replies()` - 6 edges
9. `inspect_full_csv()` - 6 edges
10. `SemanticSupportAgent` - 6 edges

## Surprising Connections (you probably didn't know these)
- `LLM-as-Judge Rubric` --semantically_similar_to--> `Escalation Labeling Rules`  [INFERRED] [semantically similar]
  Hiver SDE Intern Assignment.md → evaluation/intent_guide.md
- `Intent Classification Task` --conceptually_related_to--> `AppleSupport Intent Taxonomy (11 labels)`  [INFERRED]
  Hiver SDE Intern Assignment.md → evaluation/intent_guide.md
- `Escalation Decision Task` --conceptually_related_to--> `Escalation Labeling Rules`  [INFERRED]
  Hiver SDE Intern Assignment.md → evaluation/intent_guide.md
- `Golden Evaluation Set (150-250 examples)` --conceptually_related_to--> `Brand Support Agent Project`  [INFERRED]
  Hiver SDE Intern Assignment.md → README.md
- `Customer Support on Twitter Dataset (Kaggle ~3M tweets)` --conceptually_related_to--> `AppleSupport Brand (106k outbound replies)`  [INFERRED]
  Hiver SDE Intern Assignment.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Core Agent Pipeline (Classify + Draft + Escalate)** — hiver_sde_intern_assignment_intent_classification, hiver_sde_intern_assignment_reply_drafting, hiver_sde_intern_assignment_escalation_decision [EXTRACTED 1.00]
- **Evaluation Deliverables (Golden Set + LLM Judge + Harness)** — hiver_sde_intern_assignment_golden_eval_set, hiver_sde_intern_assignment_llm_judge, hiver_sde_intern_assignment_failure_analysis, hiver_sde_intern_assignment_decision_log [EXTRACTED 1.00]
- **High-Risk Intents (always escalate)** — evaluation_intent_guide_account_access_security, evaluation_intent_guide_app_store_billing, evaluation_intent_guide_escalation_rules [EXTRACTED 1.00]

## Communities (9 total, 0 thin omitted)

### Community 0 - "Intent Taxonomy & Escalation Rules"
Cohesion: 0.10
Nodes (27): Intent: account_access_security (always escalate), Intent: app_store_billing (always escalate), Intent: battery_power, Escalation Labeling Rules, AppleSupport Intent Taxonomy (11 labels), Intent: ios_software_bug, Intent: other_unclear (fallback), Intent Tie-Breaker Rules (+19 more)

### Community 1 - "Banking77 Preprocessing Benchmark"
Cohesion: 0.14
Nodes (21): ndarray, main(), parse_args(), Namespace, Benchmark text preprocessing on the optional Banking77 intent dataset., evaluate_variant(), main(), notebook_style_text() (+13 more)

### Community 2 - "Core Agent & Intent Classifier"
Cohesion: 0.14
Nodes (15): AgentResult, classify_intent(), DataFrame, A transparent retrieval baseline for the AppleSupport agent., SupportAgent, main(), parse_args(), Namespace (+7 more)

### Community 3 - "Data Inspection & Brand Selection"
Cohesion: 0.22
Nodes (14): inspect_full_csv(), is_brand_account(), load_csv(), main(), parse_args(), print_basic_report(), DataFrame, Namespace (+6 more)

### Community 4 - "Brand Extraction Pipeline"
Cohesion: 0.29
Nodes (12): build_pairs(), collect_brand_replies(), collect_customer_tweets(), is_outbound(), main(), parse_args(), DataFrame, Namespace (+4 more)

### Community 5 - "Topic Profiling & Clustering"
Cohesion: 0.36
Nodes (7): clean_text(), main(), parse_args(), Namespace, Series, Profile a brand-specific customer/reply table to guide intent design., safe_console_text()

### Community 6 - "Evaluation Harness"
Cohesion: 0.47
Nodes (5): main(), parse_args(), parse_gold_bool(), Namespace, Evaluate the retrieval baseline against a reviewed golden set.

### Community 7 - "Golden Set Labeling"
Cohesion: 0.50
Nodes (4): main(), parse_args(), Namespace, Create a deterministic human-labeling sheet from brand pairs.

## Knowledge Gaps
- **9 isolated node(s):** `Reply Drafting Task`, `Failure Analysis Report Section`, `Decision Log (10-15 bullets)`, `Reproducibility Target (<15 min single command)`, `Intent: ios_software_bug` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 46 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `normalize_for_classification()` connect `Banking77 Preprocessing Benchmark` to `Core Agent & Intent Classifier`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `normalize_for_classification()` (e.g. with `.__init__()` and `main()`) actually correct?**
  _`normalize_for_classification()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Reply Drafting Task`, `Failure Analysis Report Section`, `Decision Log (10-15 bullets)` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Intent Taxonomy & Escalation Rules` be split into smaller, more focused modules?**
  _Cohesion score 0.09686609686609686 - nodes in this community are weakly interconnected._
- **Should `Banking77 Preprocessing Benchmark` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._
- **Should `Core Agent & Intent Classifier` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._