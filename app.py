import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import math
from collections import Counter
from yarl import URL

# Load model
bundle = joblib.load("malicious_url_model.pkl")

model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]


# Feature Engineering
def normalize_url(url):
    url = str(url).strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url


def shannon_entropy(s):
    if not s:
        return 0
    probs = [v / len(s) for v in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)


def feature_ext(stringUrl):
    raw_url = normalize_url(stringUrl)

    try:
        url = URL("http://" + raw_url)

        host = url.host or ""
        path = url.path or ""
        query = url.query_string or ""

        has_ip = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", host) else 0
        num_subdomains = host.count(".") - 1 if host.count(".") > 1 else 0

        features = {
            "url_len": len(raw_url),
            "host_len": len(host),
            "path_len": len(path),
            "query_len": len(query),
            "has_ip": has_ip,
            "num_subdomains": num_subdomains,
            "is_std_port": 0 if url.port and url.port not in [80, 443] else 1,
            "count_dot": raw_url.count("."),
            "count_hyphen": raw_url.count("-"),
            "count_slash": raw_url.count("/"),
            "count_question": raw_url.count("?"),
            "count_equal": raw_url.count("="),
            "count_percent": raw_url.count("%"),
            "count_at": raw_url.count("@"),
            "count_digits": sum(c.isdigit() for c in raw_url),
            "entropy": shannon_entropy(raw_url),
        }

        features["digit_ratio"] = (
            features["count_digits"] / features["url_len"] if features["url_len"] else 0
        )
        features["path_ratio"] = (
            features["path_len"] / features["url_len"] if features["url_len"] else 0
        )
        features["query_ratio"] = (
            features["query_len"] / features["url_len"] if features["url_len"] else 0
        )

        return pd.Series(features)

    except Exception:
        return pd.Series({col: 0 for col in feature_columns})


# Prediction
def evaluate_url(test_url):
    struct_features = feature_ext(test_url)

    X_eval = pd.DataFrame([struct_features])
    X_eval = X_eval[feature_columns]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_eval)[0, 1]
    else:
        decision = model.decision_function(X_eval)[0]
        proba = 1 / (1 + np.exp(-decision))

    pred = int(proba >= threshold)

    return proba, pred


# Streamlit UI
st.set_page_config(page_title="Malicious URL Detector")

st.title("Malicious URL Detection")
st.write("Enter a URL to check whether it is **benign or malicious**.")

url_input = st.text_input("URL", placeholder="example.com/login/update")

if st.button("Analyze URL"):

    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Analyzing..."):
            proba, pred = evaluate_url(url_input)

        if pred == 1:
            st.error(f"Malicious ({proba:.2%} confidence)")
        else:
            st.success(f"Benign ({1-proba:.2%} confidence)")

        st.caption(f"Decision threshold: {threshold:.2f}")
