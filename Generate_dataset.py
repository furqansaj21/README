"""
generate_dataset.py
-------------------
Generates a synthetic but realistic dataset of phishing and legitimate URLs.
In a real project you would replace this with the UCI Phishing Dataset or
Kaggle's Web Page Phishing Detection dataset.

Legitimate dataset sources for your portfolio README:
  - https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
  - https://archive.ics.uci.edu/ml/datasets/phishing+websites
  - PhishTank API: https://www.phishtank.com/developer_info.php

Author : Mohammed Furqan Sajid
"""

import random
import pandas as pd
from feature_extractor import extract_features, FEATURE_NAMES

random.seed(42)

# ── sample URL pools ──────────────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com/search?q=python",
    "https://stackoverflow.com/questions/1234",
    "https://github.com/user/repo",
    "https://docs.python.org/3/library/os.html",
    "https://www.youtube.com/watch?v=abc123",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.amazon.in/dp/B08N5WRWNW",
    "https://mail.google.com/mail/u/0/",
    "https://www.linkedin.com/in/furqansajid",
    "https://www.microsoft.com/en-us/windows",
    "https://www.apple.com/iphone/",
    "https://www.bbc.com/news/technology",
    "https://www.reddit.com/r/cybersecurity/",
    "https://tryhackme.com/dashboard",
    "https://portal.azure.com/",
    "https://console.aws.amazon.com/",
    "https://cloud.google.com/security",
    "https://www.coursera.org/learn/cybersecurity",
    "https://www.ibm.com/security",
    "https://www.cisco.com/c/en/us/products/security",
    "https://naukri.com/cyber-security-jobs",
    "https://www.infosys.com/services/cyber-security",
    "https://www.tcs.com/cybersecurity-services",
    "https://splunk.com/en_us/products/splunk-enterprise.html",
    "https://www.wireshark.org/docs/",
]

PHISHING_URLS = [
    "http://192.168.1.1/login/verify-account.php?user=admin",
    "http://paypal.com.secure-verify.attacker.com/signin",
    "http://www.payp-al.com/secure/login?cmd=_session",
    "http://amazon-update.com/account/login?ref=suspicious",
    "http://login-verify-secure.banking-update.net/auth",
    "http://216.58.214.100/login.php?redirect=paypal",
    "http://secure-account-verify.com/login?session=abc123def456",
    "http://ebay.com.account-suspended.badsite.ru/restore",
    "http://apple-id-verify-account-secure.com/signin",
    "http://microsoft.com.update-required.phishsite.com/verify",
    "http://www.free-winner-bonus.com/claim?user=victim",
    "http://bankofamerica-secure.com/login:8080/auth.php",
    "http://support-google-verify.com/account-recovery",
    "http://facebook-security-alert.com/confirm-identity",
    "http://instagram.com@phishing-page.com/login",
    "http://verify-your-account-immediately.com/update",
    "http://amazon.co.uk.account-suspended12345.xyz/restore",
    "http://credential-update-required.net/banking/login",
    "http://173.194.122.100/paypal/login?secure=1",
    "http://confirm-email-password-reset.com/secure/auth",
    "http://www.lucky-winner-2026.com/claim-prize?id=1234",
    "http://signin.secure-paypal-verify.com/webscr?cmd=login",
    "http://apple-support-account-locked.com/verify-now",
    "http://update-your-banking-info.com/login:9090/",
    "http://www.microsoft-alert-account.com/support/verify",
]


def augment_urls(base_urls: list, n: int, is_phishing: bool) -> list:
    """Create variations of base URLs to reach desired count n."""
    augmented = list(base_urls)
    suffixes = ["/home", "/index.php", "/dashboard", "/profile", "/settings"]
    params   = ["?id=", "?ref=", "?session=", "?token=", "?redirect="]

    while len(augmented) < n:
        base = random.choice(base_urls)
        variation = base + random.choice(suffixes) + random.choice(params) + str(random.randint(1000, 9999))
        augmented.append(variation)

    return augmented[:n]


def build_dataset(n_legit: int = 500, n_phish: int = 500) -> pd.DataFrame:
    """
    Build a balanced dataset of legitimate and phishing URLs.

    Parameters
    ----------
    n_legit : int  Number of legitimate URL samples
    n_phish : int  Number of phishing URL samples

    Returns
    -------
    pd.DataFrame  Columns: FEATURE_NAMES + ['label']  (0=legit, 1=phish)
    """
    legit_urls = augment_urls(LEGITIMATE_URLS, n_legit, is_phishing=False)
    phish_urls = augment_urls(PHISHING_URLS,   n_phish, is_phishing=True)

    all_urls   = legit_urls + phish_urls
    all_labels = [0] * n_legit + [1] * n_phish

    rows = []
    for url, label in zip(all_urls, all_labels):
        features = extract_features(url)
        rows.append(features + [label])

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    return df


if __name__ == "__main__":
    df = build_dataset(500, 500)
    df.to_csv("phishing_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['label'].value_counts().to_dict()}")
    print(df.head())
