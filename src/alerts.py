# src/alerts.py

def check_early_warning(features: dict) -> list[str]:
    warnings = []

    if features.get("trestbps", 0) > 160:
        warnings.append("⚠️ Resting blood pressure is > 160 mm Hg. Please seek urgent medical review.")

    if features.get("chol", 0) > 300:
        warnings.append("⚠️ Cholesterol > 300 mg/dL. Consider immediate lipid-lowering therapy.")

    if features.get("age", 0) > 75:
        warnings.append("⚠️ Age > 75. Schedule regular cardiology check-ups.")

    return warnings

if __name__ == "__main__":
    sample = {"age": 78, "trestbps": 165, "chol": 320}
    w = check_early_warning(sample)
    print("Warnings:", w)
