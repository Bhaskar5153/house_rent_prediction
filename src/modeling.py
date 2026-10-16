import argparse
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from xgboost import XGBRegressor

from src.config import load_config
from src.preprocessing import preprocess_house_data
from src.outlier_handler import remove_outlier_iqr

import ipdb
import joblib
from pathlib import Path



NUMERICAL_FEATURES = ['house_size', 'latitude', 'longitude', 'price', 'num_bathrooms',
       'num_balconies', 'is_negotiable', 'price_sqft', 'security_deposite',
       'bhk', 'verification_days', 'is_furnished', 'is_semi_furnished',
       'is_unfurnished', 'listing_age_days', 'has_parking', 'has_garden',
       'has_cctv', 'has_power_backup', 'near_hospital', 'near_metro',
       'area_per_room', 'bathrooms_per_room']

CATEGORICAL_FEATURES = ['house_type', 'location', 'city', 'verification_date', 'description',
       'Status', 'source_file', 'property_type', 'status']


columns_to_drop = [
    "status", "Status", "source_file", "property_type", "price_sqft", "security_deposite", "verification_days","bathrooms_per_room"
]

NUMERICAL_FEATURES = [col for col in NUMERICAL_FEATURES if col not in columns_to_drop]
CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col not in columns_to_drop]



def training_code(data, config):
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data.city)
    train = train.drop(columns=columns_to_drop, errors="ignore")
    test = test.drop(columns=columns_to_drop, errors="ignore")
    clean_train = remove_outlier_iqr(train, ['house_size'], multiplier=config.outlier_iqr_multiplier, group_by='city')



    numeric = SimpleImputer(strategy='median', keep_empty_features=True)
    categorical = Pipeline(
        [
            ('fill_missing', SimpleImputer(strategy='constant', fill_value='unknown', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )

    prepare = ColumnTransformer(
        [
            ('numeric', numeric, NUMERICAL_FEATURES),
            ('categorical', categorical, CATEGORICAL_FEATURES)
        ]
    )

    model = Pipeline(
        [
            ('prepare', prepare),
            ('model', XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.05, objective='reg:squarederror', n_jobs=2, random_state=42))
            
        ]
    )

    model.fit(clean_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], clean_train[config.target])
    # ipdb.set_trace()





    return {
        "model": model, "train": train, "clean_train": clean_train,
    }



def predict_raw(model, raw):
    return model.predict(preprocess_house_data(raw, training=False))
# ipdb.set_trace()


def load_processed_data(path, target="price"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} is missing")

    data = pd.read_csv(path)

    required_columns = list(dict.fromkeys(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [target]))

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Processed.csv is missing columns: {missing}. run preprocessing.py first")
    return data
# ipdb.set_trace()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Processed.csv: default available in artifacts")
    parser.add_argument("--config", default="configs/config.yml")
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(args.input) if args.input else config.artifact_dir / 'processed.csv'
    data = load_processed_data(source, config.target)
    result = training_code(data, config)

    result['data_source'] = str(source.resolve())

    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], model_path)
    print(f"Model saved to {model_path}")


    # ipdb.set_trace()
    
    print(result)


if __name__ == "__main__":
    main()





















