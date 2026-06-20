import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from utils import suppress_c_stderr

def main():
    model_path = "models/mlp_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/label_encoder.pkl"

    print("Checking if files exist...")
    for path in [model_path, scaler_path, encoder_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            sys.exit(1)

    print("Loading model, scaler, and label encoder...")
    try:
        with suppress_c_stderr():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Label encoder classes: {list(label_encoder.classes_)}")
    print(f"Model type: {type(model).__name__}")

    n_features = model.coefs_[0].shape[0]
    print(f"Input features: {n_features}")

    print("Performing dummy inference...")
    try:
        dummy_input = np.random.rand(1, n_features).astype(np.float32)
        dummy_input_scaled = scaler.transform(dummy_input)
        preds = model.predict_proba(dummy_input_scaled)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        label = label_encoder.inverse_transform([class_idx])[0]
        print(f"Inference test passed! Predicted class: {label} with confidence {confidence:.2f}")
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

    print("All checks passed successfully!")

if __name__ == "__main__":
    main()
