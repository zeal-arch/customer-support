import sys
sys.path.insert(0, ".")
import pandas as pd
from src.intent_classifier import IntentClassifier, build_from_pairs

golden = pd.read_csv("evaluation/golden_set.csv")

# Create clean audit according to evaluation/intent_guide.md
audit_df = golden.copy()

# Fix known regex-collision errors in ground truth:
# Row 30: locked out of Apple ID -> account_access_security (Esc: True)
audit_df.loc[30, "intent"] = "account_access_security"
audit_df.loc[30, "gold_escalate"] = True
audit_df.loc[30, "label_source"] = "expert_audit"

# Row 40: 403 error on dev portal account -> app_service_issue (Esc: True)
audit_df.loc[40, "intent"] = "app_service_issue"
audit_df.loc[40, "gold_escalate"] = True
audit_df.loc[40, "label_source"] = "expert_audit"

# Row 52: sync via iTunes Match / iBooks -> setup_transfer_sync (Esc: False - standard sync troubleshooting)
audit_df.loc[52, "intent"] = "setup_transfer_sync"
audit_df.loc[52, "gold_escalate"] = False
audit_df.loc[52, "label_source"] = "expert_audit"

# Row 77: watch % drop within minutes -> battery_power (Esc: False)
audit_df.loc[77, "intent"] = "battery_power"
audit_df.loc[77, "gold_escalate"] = False
audit_df.loc[77, "label_source"] = "expert_audit"

# Row 92: mac pro won't read sd card -> hardware_accessory (Esc: False)
audit_df.loc[92, "intent"] = "hardware_accessory"
audit_df.loc[92, "gold_escalate"] = False
audit_df.loc[92, "label_source"] = "expert_audit"

# Row 131: Activity issues with my Watch -> app_service_issue (Esc: False)
audit_df.loc[131, "intent"] = "app_service_issue"
audit_df.loc[131, "gold_escalate"] = False
audit_df.loc[131, "label_source"] = "expert_audit"

# Row 138: why can't I edit or share this picture -> app_service_issue (Esc: False)
audit_df.loc[138, "intent"] = "app_service_issue"
audit_df.loc[138, "gold_escalate"] = False
audit_df.loc[138, "label_source"] = "expert_audit"

# Row 157: what charger do I need to buy -> hardware_accessory (Esc: False)
audit_df.loc[157, "intent"] = "hardware_accessory"
audit_df.loc[157, "gold_escalate"] = False
audit_df.loc[157, "label_source"] = "expert_audit"

# Row 158: keeps charging me $1 a month for icloud storage -> app_store_billing (Esc: True)
audit_df.loc[158, "intent"] = "app_store_billing"
audit_df.loc[158, "gold_escalate"] = True
audit_df.loc[158, "label_source"] = "expert_audit"

# Row 197: 3rd time I've had to charge my iPhone -> battery_power (Esc: False)
audit_df.loc[197, "intent"] = "battery_power"
audit_df.loc[197, "gold_escalate"] = False
audit_df.loc[197, "label_source"] = "expert_audit"

# Row 220: podcast app changed, subscriptions disappeared -> app_service_issue (Esc: False)
audit_df.loc[220, "intent"] = "app_service_issue"
audit_df.loc[220, "gold_escalate"] = False
audit_df.loc[220, "label_source"] = "expert_audit"

# Row 223: phishing text message looks fake -> account_access_security (Esc: True)
audit_df.loc[223, "intent"] = "account_access_security"
audit_df.loc[223, "gold_escalate"] = True
audit_df.loc[223, "label_source"] = "expert_audit"

# Row 29: battery has been draining after 11.1.1 update -> battery_power (Esc: False)
audit_df.loc[29, "intent"] = "battery_power"
audit_df.loc[29, "gold_escalate"] = False
audit_df.loc[29, "label_source"] = "expert_audit"

# Row 182: battery duration is a damn joke -> battery_power (Esc: False)
audit_df.loc[182, "intent"] = "battery_power"
audit_df.loc[182, "gold_escalate"] = False
audit_df.loc[182, "label_source"] = "expert_audit"

# Row 44: lose over 90k of my pics -> app_service_issue (Esc: True)
audit_df.loc[44, "intent"] = "app_service_issue"
audit_df.loc[44, "gold_escalate"] = True
audit_df.loc[44, "label_source"] = "expert_audit"

# Row 6: "FIX IT" keyboard autocorrect glitch -> ios_software_bug (Esc: False)
audit_df.loc[6, "intent"] = "ios_software_bug"
audit_df.loc[6, "gold_escalate"] = False
audit_df.loc[6, "label_source"] = "expert_audit"

# Row 39: bought itunes giftcard in pakistan -> app_store_billing (Esc: True)
audit_df.loc[39, "intent"] = "app_store_billing"
audit_df.loc[39, "gold_escalate"] = True
audit_df.loc[39, "label_source"] = "expert_audit"

# Row 231: punched my computer really hard and it has a dent -> hardware_accessory (Esc: False)
audit_df.loc[231, "intent"] = "hardware_accessory"
audit_df.loc[231, "gold_escalate"] = False
audit_df.loc[231, "label_source"] = "expert_audit"

print("Audited intents value counts:")
print(audit_df["intent"].value_counts())

# Save clean audited golden set
audit_df.to_csv("evaluation/golden_set_audited.csv", index=False)
print("Saved evaluation/golden_set_audited.csv")
