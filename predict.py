"""
predict.py
----------
Load the trained model and analyse any URL for phishing probability.
Can be used as:
  - Interactive CLI tool  (python predict.py)
  - Imported as a module  (from predict import predict_url)

Author : Mohammed Furqan Sajid
Project: Phishing URL Detection using Machine Learning
"""

import os
import pickle
import sys

from feature_extractor import extract_features, FEATURE_NAMES


# ── constants ─────────────────────────────────────────────────────────────────

MODEL_PATH  = "phishing_model.pkl"
SCALER_PATH = "scaler.pkl"

RISK_LEVELS = {
    (0.00, 0.30): ("LOW",      "✅", "Likely legitimate"),
    (0.30, 0.60): ("MEDIUM",   "⚠️ ", "Suspicious — verify manually"),
    (0.60, 0.80): ("HIGH",     "🔴", "Likely phishing — do not click"),
    (0.80, 1.01): ("CRITICAL", "🚨", "Almost certainly phishing"),
}


# ── load model ────────────────────────────────────────────────────────────────

def load_model():
    """Load the saved model and scaler. Raises FileNotFoundError if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Please run:  python train_model.py  first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ── core prediction ───────────────────────────────────────────────────────────

def predict_url(url: str, model=None, scaler=None) -> dict:
    """
    Predict whether a URL is phishing or legitimate.

    Parameters
    ----------
    url    : str   The URL to analyse.
    model  : fitted RandomForestClassifier (optional — loaded from disk if None)
    scaler : fitted StandardScaler         (optional — loaded from disk if None)

    Returns
    -------
    dict with keys:
        url, prediction, phishing_probability, risk_level,
        risk_emoji, risk_message, features
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    features      = extract_features(url)
    features_scaled = scaler.transform([features])
    prediction    = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    phish_prob    = probabilities[1]

    # determine risk level
    risk_level = risk_emoji = risk_message = ""
    for (lo, hi), (level, emoji, msg) in RISK_LEVELS.items():
        if lo <= phish_prob < hi:
            risk_level   = level
            risk_emoji   = emoji
            risk_message = msg
            break

    return {
        "url"                 : url,
        "prediction"          : "PHISHING" if prediction == 1 else "LEGITIMATE",
        "phishing_probability": round(phish_prob * 100, 2),
        "legit_probability"   : round(probabilities[0] * 100, 2),
        "risk_level"          : risk_level,
        "risk_emoji"          : risk_emoji,
        "risk_message"        : risk_message,
        "features"            : dict(zip(FEATURE_NAMES, features)),
    }


# ── display helpers ───────────────────────────────────────────────────────────

def print_result(result: dict, verbose: bool = False):
    """Pretty-print a prediction result to the terminal."""
    print("\n" + "─" * 60)
    print(f"  URL      : {result['url'][:80]}")
    print(f"  Result   : {result['risk_emoji']}  {result['prediction']}  —  {result['risk_message']}")
    print(f"  Phishing : {result['phishing_probability']:.1f}%  |  Legitimate: {result['legit_probability']:.1f}%")
    print(f"  Risk     : {result['risk_level']}")

    if verbose:
        print("\n  Feature breakdown:")
        for name, val in result["features"].items():
            flag = "  ▶" if val > 0 and name not in ("url_length", "count_dots",
                                                       "path_length", "count_special_chars",
                                                       "domain_length", "count_digits_in_domain") else "   "
            print(f"    {flag} {name:<35} {val}")
    print("─" * 60)


# ── batch analysis ────────────────────────────────────────────────────────────

def analyse_batch(urls: list, model=None, scaler=None) -> list:
    """
    Analyse a list of URLs and return a list of result dicts.
    Useful for CSV/file-based scanning.
    """
    if model is None or scaler is None:
        model, scaler = load_model()
    return [predict_url(url, model, scaler) for url in urls]


# ── interactive CLI ───────────────────────────────────────────────────────────

def interactive_mode():
    """Run an interactive command-line scanner."""
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("   Phishing URL Detector — Mohammed Furqan Sajid")
    print("   Random Forest | 14 URL features | SOC Portfolio")
    print("=" * 60)
    print("Commands:  'quit' to exit  |  'verbose' to toggle details\n")

    verbose = False

    while True:
        try:
            raw = input("Enter URL > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() == "quit":
            print("Goodbye.")
            break
        if raw.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        # auto-add scheme if missing so urllib.parse works correctly
        url = raw if raw.startswith("http") else "http://" + raw

        try:
            result = predict_url(url, model, scaler)
            print_result(result, verbose=verbose)
        except Exception as ex:
            print(f"[ERROR] Could not analyse URL: {ex}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # single URL passed as command-line argument
        url = sys.argv[1]
        if not url.startswith("http"):
            url = "http://" + url
        try:
            model, scaler = load_model()
            result = predict_url(url, model, scaler)
            print_result(result, verbose=True)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
    else:
        interactive_mode()
