# Decision Log

Non-obvious decisions made during this project and why.

---

1. **Brand: AppleSupport over all others**
   Raw scan found AppleSupport had the highest outbound reply count (106,860) among all support accounts in the dataset. Higher volume -> richer retrieval corpus -> more likely to find relevant historical examples. Could have picked Amazon or Spotify but AppleSupport also has diverse, well-scoped support issues (hardware, software, billing, security).

2. **11 intents, not fewer and not 77**
   Banking77's 77-label schema is too granular for a 106k-tweet corpus where many categories would have < 100 examples. A 3-5 label schema loses too much specificity (hardware and billing conflated is dangerous). 11 labels emerged from two rounds of KMeans clustering on the corpus and match the natural topic groups observed.

3. **Conservative text preprocessing (URL -> URL token, @mention -> USER token; keep everything else)**
   The notebook-style preprocessing (remove punctuation, stopwords) achieves higher retrieval similarity scores but destroys signal that matters for support: product names (`AirPods`, `iOS 17`), complaint intensity (`STILL broken`), and technical terms (`FaceTime`, `AirDrop`). The conservative approach sacrifices 10 pp of retrieval similarity score but preserves classifiable signal.

4. **TF-IDF + Logistic Regression, not a transformer**
   8 GB RAM / 4 GB VRAM rules out fine-tuning BERT locally. TF-IDF + LR fits comfortably in RAM, trains in under 2 minutes on 18k examples, and is fully transparent -- you can inspect which n-grams drive each prediction. For the submission's explainability requirement, this is strictly better than a black-box model.

5. **Weak supervision for classifier training (rule-based classifier -> LR training data)**
   We have no human-labeled training set for the 11 intents. Instead, we apply the rule-based regex classifier to 40k pairs, keep high-confidence predictions (>= 0.60), and use those as noisy training labels. This is a standard self-training pattern; its limitation is that the LR model learns the rule-based classifier's biases, which is documented in the "What is misleading" section.

6. **Hard escalation rules for account_access_security and app_store_billing**
   These two intents carry financial and security risk. Any automated reply that handles them wrongly is worse than no reply at all. Hard escalation is the conservative-correct choice regardless of confidence or retrieval quality.

7. **Soft escalation on confidence < 0.45 and similarity < 0.20**
   These thresholds were set by inspecting the distribution of LR confidence scores and TF-IDF cosine similarities on a development sample. Below 0.45 confidence the most likely predicted class changes with minor text edits, indicating unreliable classification. Below 0.20 cosine similarity the retrieved example is essentially unrelated.

8. **Leakage prevention: remove golden-set IDs from the retrieval reference**
   If we let the agent retrieve the exact example it is being evaluated on, the retrieval similarity score is artificially inflated and the draft reply is the gold standard's own text. We hold out the 250 golden tweet IDs from the reference pool before any retrieval or evaluation.

9. **Majority class for trivial baseline, not a random baseline**
   A random classifier on an imbalanced dataset (73% `ios_software_bug`) performs worse than a majority-class predictor. Using the majority class is the standard "sensible floor" for structured prediction tasks and is a harder baseline to beat.

10. **Reply = retrieved historical example, not LLM-generated text**
    LLM generation adds an uncontrolled hallucination risk and requires either a paid API or local hardware we don't have. Retrieved examples are always grounded in real AppleSupport responses, which is a stronger safety guarantee for a support agent. The trade-off is that replies are sometimes slightly off-topic (when retrieval quality is low).

11. **250 golden set examples, not 150**
    The assignment specifies 150-250. 250 gives more statistical power for macro-F1 (rare classes like `setup_transfer_sync` need multiple examples to show up in precision/recall). Sampling more costs labeling time but we automated labeling, making 250 essentially free.

12. **Two-pass labeling: rule-based first, LR second for low-confidence rows**
    The rule-based classifier is precise on its strongest patterns (battery, security, billing) but weak on ambiguous messages (classified as `other_unclear`). The LR pass recovers meaningful labels for 126 of those rows, reducing the `other_unclear` rate from 40% to near 0%.

13. **Optional Banking77 experiment kept separate from AppleSupport evaluation**
    Banking77 has 77 domain-specific banking intents. It is useful for benchmarking preprocessing choices (do URLs/mentions removal help or hurt?) but its labels are incompatible with our 11-intent schema. Results are stored separately and clearly labelled as "not AppleSupport performance."

14. **MiniLM semantic backend kept optional**
    `sentence-transformers/all-MiniLM-L6-v2` produces better retrieval quality than TF-IDF but requires ~380 MB of model weights and longer inference time. On the hardware constraint (8 GB RAM, no GPU), it is usable but noticeably slower. We keep it as `--backend minilm` so a grader with better hardware can compare.

15. **One-file-per-module structure in src/**
    Each step of the pipeline (inspect → extract → profile → classify → retrieve → evaluate) is a separate module runnable with `python -m src.<module>`. This makes the pipeline transparent, testable in isolation, and reproducible without a notebook kernel.
