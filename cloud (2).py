import os
import math
import string
import json
from collections import Counter
import numpy as np
import pandas as pd
import joblib
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB

# ---------------------------
# Feature Extraction
# ---------------------------

def load_binary(path):
    with open(path, "rb") as f:
        return f.read()

def entropy(data):
    if not data:
        return 0.0
    counter = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def byte_histogram(data):
    histogram = [0] * 256
    for b in data:
        histogram[b] += 1
    total = len(data)
    return [round(h / total, 5) for h in histogram] if total > 0 else [0] * 256

def extract_strings(data, min_length=4):
    result = []
    current = ''
    for byte in data:
        c = chr(byte)
        if c in string.printable and c != '\x00':  # Allow spaces, tabs, etc., but not null bytes
            current += c
        else:
            if len(current) >= min_length:
                result.append(current)
            current = ''
    if len(current) >= min_length:
        result.append(current)
    return result

def extract_byte_ngrams(data, n=2, top_k=10):
    ngrams = [data[i:i+n] for i in range(len(data) - n + 1)]
    freq = Counter(ngrams)
    total = sum(freq.values())
    normalized = {k: round(v / total, 5) for k, v in freq.items()} if total > 0 else {}
    top = dict(Counter(normalized).most_common(top_k))
    return {f'ngram_{k.hex()}': v for k, v in top.items()}

def extract_opcodes(data, base_addr=0x08000000):
    try:
        md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
        instructions = md.disasm(data, base_addr)
        opcodes = [insn.mnemonic for insn in instructions]
        freq = Counter(opcodes)
        return {f'op_{op}': freq[op] for op in freq}
    except Exception:
        return {}

def extract_features_from_bin(file_path):
    data = load_binary(file_path)
    features = {
        "file_name": os.path.basename(file_path),
        "size": len(data),
        "entropy": entropy(data)
    }

    # Byte histogram
    byte_hist = byte_histogram(data)
    features.update({f'byte_{i}': val for i, val in enumerate(byte_hist)})

    # Strings
    strings = extract_strings(data)
    string_count = len(strings)
    total_len = sum(len(s) for s in strings)
    features["string_count"] = string_count
    features["avg_string_len"] = total_len / string_count if string_count > 0 else 0
    features["max_string_len"] = max((len(s) for s in strings), default=0)
    features["min_string_len"] = min((len(s) for s in strings), default=0)
    features["total_string_len"] = total_len

    # Byte N-grams
    features.update(extract_byte_ngrams(data, n=2, top_k=10))

    # Opcodes
    features.update(extract_opcodes(data))

    return features, data, strings

# ---------------------------
# Prediction
# ---------------------------

def preprocess_features(features, expected_columns):
    df = pd.DataFrame([features])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    return df.to_numpy()

def get_next_version_number():
    """Determine the next version number by checking existing firmware_v*.json files."""
    version = 1
    while os.path.exists(f"firmware_v{version}.json") or os.path.exists(f"firmware_v{version}.bin"):
        version += 1
    return version

def test_model_on_bin(model_path, encoder_path, features_csv_path):
    # Load model and label encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Get expected feature columns
    df_ref = pd.read_csv(features_csv_path, nrows=1)
    expected_columns = [col for col in df_ref.columns if col not in ("file_name", "label")]

    while True:
        # Prompt user for binary file
        file_path = input("\n📂 Enter path to .bin file (or 'quit' to exit): ").strip()
        if file_path.lower() in ["quit", ""]:
            print("👋 Exiting program.")
            break
        if not os.path.isfile(file_path):
            print("❌ File not found. Please try again.")
            continue

        # Get next version number
        version = get_next_version_number()

        # Extract features, binary data, and strings
        features, binary_data, strings = extract_features_from_bin(file_path)
        input_data = preprocess_features(features, expected_columns)

        # Predict
        prediction = model.predict(input_data)
        label = label_encoder.inverse_transform(prediction)[0]
        print(f"\n🎯 Prediction: {label.upper()}")

        probs = model.predict_proba(input_data)[0]
        for i, prob in enumerate(probs):
            print(f"   {label_encoder.classes_[i]}: {prob:.4f}")

        # Create JSON file with metadata
        json_data = {
            "file_name": os.path.basename(file_path),
            "size": len(binary_data),
            "strings": strings,
            "label": label
        }
        json_filename = f"firmware_v{version}.json"
        with open(json_filename, "w") as f:
            json.dump(json_data, f, indent=4)

        # Save the input binary file
        bin_filename = f"firmware_v{version}.bin"
        with open(bin_filename, "wb") as f:
            f.write(binary_data)

        print(f"\n📄 Generated '{json_filename}' with metadata.")
        print(f"📦 Saved input binary as '{bin_filename}'.")

# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    test_model_on_bin(
        model_path="malware_detector_model.pkl",
        encoder_path="label_encoder.pkl",
        features_csv_path="features.csv"
    )

