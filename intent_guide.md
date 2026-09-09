# AppleSupport intent guide

Use exactly one primary intent per customer message. Label the customer's underlying support need, not the tone of the message or the wording of AppleSupport's reply.

## Labels

| Label | Use when | Examples of signals |

## ----------------------------------------------------------------------------------------------------------------------------------------------------

| `ios_software_bug` | The customer reports an iOS or built-in software bug, crash, freeze, keyboard problem, update regression, or unexpected device behavior. | "question mark boxes", "after iOS 11", "keeps freezing", "autocorrect is broken" |
| `battery_power` | The main problem is battery life, charging, power, overheating, or a device that will not turn on. | "battery drains", "won't charge", "battery health" |
| `app_service_issue` | A specific Apple app or service is the problem, excluding App Store billing. | Photos, Music, Messages, Maps, Safari, iCloud, Apple Pay, FaceTime |
| `account_access_security` | Apple ID, password, login, account lockout, disabled account, or suspected account compromise. | "forgot password", "Apple ID disabled", "someone accessed my account" |
| `setup_transfer_sync` | Setting up a device, restoring a backup, moving data, or syncing devices/accounts. | "new iPhone", "transfer data", "restore backup", "iCloud sync" |
| `app_store_billing` | App Store purchases, subscriptions, refunds, payment methods, or unexpected charges. | "cancel subscription", "charged twice", "refund an app" |
| `connectivity_network` | Wi-Fi, cellular data, Bluetooth, AirDrop, Handoff, SIM, or connection problems. | "won't connect", "Wi-Fi drops", "Bluetooth pairing" |
| `hardware_accessory` | Physical device damage, display, buttons, camera, headphones, chargers, adapters, or replacement/repair. | "cracked screen", "headphones not detected", "replace battery" |
| `how_to_settings` | The customer asks how to use a feature or change a setting without reporting a failure. | "how do I...", "where can I enable...", "can I change..." |
| `complaint_feedback` | General dissatisfaction, product feedback, or criticism without a clear actionable support issue. | "this update is terrible", "Apple needs to fix..." with no specific issue |
| `other_unclear` | The message is too short, image-only, unrelated, or cannot be assigned confidently. | "help", "yes", "see image", "thanks" |

## Escalation labeling

Set `gold_escalate` to `true` when a human should review the case before the agent answers. Escalate when the message is ambiguous, requires account-specific investigation, involves security/fraud, legal or safety concerns, a refund/charge dispute, or has no safe evidence-backed answer. Routine how-to questions with a clear historical resolution can be `false`.

## Tie-breakers

- Choose the specific problem over `complaint_feedback` when a concrete issue is present.
- Choose `app_store_billing` over `app_service_issue` when money, subscription, purchase, or refund is central.
- Choose `setup_transfer_sync` over `how_to_settings` when moving, restoring, or syncing data is central.
- Choose `ios_software_bug` over `how_to_settings` when the customer says the feature is broken or behaves unexpectedly.
- Use `other_unclear` instead of guessing from an image link or a one-word reply.

---

## Sampling and Labeling Methodology (250 Examples)

### Sampling Strategy

The 250 evaluation examples in `golden_set.csv` were sampled from the 106,648 AppleSupport conversation pairs extracted from the Kaggle _Customer Support on Twitter_ dataset (`twcs.csv`).

- **Stratification**: To avoid an evaluation set dominated exclusively by the majority class (`ios_software_bug`), sampling was stratified across lexical clusters to ensure robust representation of rare but high-stakes categories (e.g., account security, App Store billing disputes, battery power loss, hardware accessories, and device sync).
- **Leakage Prevention**: All 250 customer tweet IDs in the golden set are strictly excluded from the agent's historical retrieval pool before any baseline evaluation runs.

### Labeling & Audit Protocol

- **Initial Pass**: Automated high-confidence lexical matching mapped unambiguous queries into candidate intents.
- **Expert Review & Audit**: Every candidate row was manually audited and finalized against the definitions and tie-breakers in this guide. Ambiguous boundary cases (such as queries mentioning both an iOS update and rapid battery depletion) were resolved in favor of the actionable primary failure mode (`battery_power`). Financial charges on cloud subscriptions were separated from device charging.
- **Escalation Ground Truth**: `gold_escalate` was set to `true` whenever an issue required account-specific credentials, involves billing/refund transactions, fraud/phishing reports, device damage, or when the message lacked sufficient diagnostic detail for an automated reply. Routine troubleshooting and configuration inquiries were set to `false`.
