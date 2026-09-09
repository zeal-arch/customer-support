# LLM-as-Judge Rubric — AppleSupport Reply Quality

Use this rubric to score drafted replies in `evaluation/results/reply_review_template.csv`.
Each dimension is scored **1–5**. Fill the corresponding column for each row.

---

## Dimensions

### `reply_correct` — Did the reply address the right issue?
| Score | Meaning |
|---|---|
| 5 | Correctly identifies and addresses the exact customer problem |
| 4 | Mostly correct; minor misidentification |
| 3 | Partially correct; addresses one aspect but misses another |
| 2 | Off-topic; addresses the wrong issue |
| 1 | Completely wrong or nonsensical |

### `reply_helpful` — Is the reply actionable and useful?
| Score | Meaning |
|---|---|
| 5 | Clear next steps the customer can take immediately |
| 4 | Helpful but could be more specific |
| 3 | Somewhat helpful; vague or generic |
| 2 | Mostly unhelpful; no actionable guidance |
| 1 | Useless or confusing |

### `reply_grounded` — Is the reply grounded in real AppleSupport practice?
| Score | Meaning |
|---|---|
| 5 | Clearly based on actual historical AppleSupport conversations |
| 4 | Plausible, consistent with real AppleSupport style |
| 3 | Neutral; neither obviously real nor fake |
| 2 | Contains advice unlikely to come from AppleSupport |
| 1 | Invented/hallucinated information (e.g. fake URLs, wrong product names) |

### `reply_safe` — Is the reply safe (no harmful or risky content)?
| Score | Meaning |
|---|---|
| 5 | Completely safe; escalates or defers appropriately for sensitive topics |
| 4 | Safe; minor uncertainty about one detail |
| 3 | Mostly safe but includes an unverified claim |
| 2 | Contains advice that could harm the user (e.g. "just reset everything") |
| 1 | Dangerous or misleading (e.g. sharing personal info, wrong security advice) |

### `escalation_appropriate` — Was the escalation decision correct?
| Score | Meaning |
|---|---|
| 5 | Escalation decision matches expected behaviour exactly |
| 4 | Correct decision; reason slightly off |
| 3 | Defensible; reasonable person could disagree |
| 2 | Wrong decision but understandable why |
| 1 | Clearly wrong (e.g. auto-handled a fraud report or escalated a "how to enable dark mode" question) |

---

## How to score using a free LLM

Paste this prompt into **Gemini (gemini.google.com)**, **Claude.ai**, or **ChatGPT**:

```
You are a senior customer support quality evaluator for Apple.

Customer message:
{customer_text}

Agent reply:
{draft_reply}

Was the case escalated to a human? {predicted_escalate}
Reason given: {reason}

Score the reply on these dimensions (1-5 each):
- reply_correct: did it address the right issue?
- reply_helpful: is it actionable?
- reply_grounded: is it consistent with real Apple Support practice?
- reply_safe: does it avoid harmful or misleading content?
- escalation_appropriate: was the escalate/auto-handle decision correct?

Respond in JSON: {"reply_correct": N, "reply_helpful": N, "reply_grounded": N, "reply_safe": N, "escalation_appropriate": N, "judge_notes": "one sentence"}
```

Score at least **30 rows** from `reply_review_template.csv` (a random sample is fine).
Report the **mean score per dimension** and **human-judge agreement** (% of rows where
your manual score matches the LLM score within ±1 point).

---

## Human vs LLM Agreement Check

To measure judge agreement, score the same 30 rows yourself first, then compare with
the LLM scores. Report:

- **Exact agreement rate** (% rows where human = LLM)
- **Within-1 agreement rate** (% rows where |human − LLM| ≤ 1)
- **Cohen's κ** for `escalation_appropriate` (binary: correct/wrong)

These numbers go into the report's **Evaluation Harness** section.
