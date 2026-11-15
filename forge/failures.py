"""Conservative rule-based root-cause hypotheses, never claimed as proven causes."""


def analyze_failure(attempt):
    text = str(attempt.get("error", "")) + str(attempt.get("check_errors", {}))
    if "Cannot construct a Series from a DataFrame" in text or "inconsistent numbers of samples: [1," in text:
        return {"category": "text_dataframe_contract", "confidence": "high", "remedy": "The host fits a DataFrame with text column. Select X['text'] in the first transformer, fill missing values, return one string per input row, and use the same pipeline input contract during prediction.", "reusable": True}
    if "No module named 'generated'" in text and "pickle" in text.lower():
        return {"category": "validator_loader", "confidence": "high", "remedy": "Register the dynamic module in sys.modules before execution; revalidate without modifying model code.", "reusable": False}
    if "unknown categor" in text.lower():
        return {"category": "unknown_category", "confidence": "high", "remedy": "Configure the encoder to support unseen categories and rerun the regression check.", "reusable": True}
    if "Plan mismatch" in text:
        return {"category": "plan_noncompliance", "confidence": "high", "remedy": "Make classifier, parameters and preprocessing match the validated plan; do not weaken checks.", "reusable": True}
    if attempt.get("failure_kind") == "timeout":
        return {"category": "resource_limit", "confidence": "medium", "remedy": "Bound model size and threads; profile before increasing time limits.", "reusable": True}
    if "model_reload" in text:
        return {"category": "serialization", "confidence": "medium", "remedy": "Inspect custom class importability and persistence behavior; isolate validator defects from model defects.", "reusable": True}
    return {"category": "unclassified", "confidence": "low", "remedy": "Inspect the preserved traceback and failed checks; do not infer a specific preprocessing remedy.", "reusable": False}
