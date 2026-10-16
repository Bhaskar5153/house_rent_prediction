import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.modeling import predict_raw


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument('--model', default='artifacts/model.joblib')
    parser.add_argument('--output', default='artifacts/predictions.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    raw['predicted_price'] = predict_raw(joblib.load(args.model), raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output, index=False)
    print(f"Saved {len(raw)} predictions to {output}")


if __name__ == "__main__":
    main()

    