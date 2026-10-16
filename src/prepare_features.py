import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame, include_target_features: bool = False) -> pd.DataFrame:
    df = df.copy()

    if include_target_features and "price" in df.columns and "house_size" in df.columns:
        df["price_per_sqft_calc"] = df.apply(
            func=lambda row: row['price'] / row['house_size'] if pd.notna(row['price']) and pd.notna(row['house_size']) and row['house_size'] > 0 else None,
            axis=1
        )


    if include_target_features and "security_deposite" in df.columns and "price" in df.columns:
        df["deposite_ratio"] = df.apply(
            lambda row: row['security_deposite'] / row['price'] if pd.notna(row['security_deposite']) and pd.notna(row['price']) and row['price'] > 0 else None,
            axis=1
        )


    if "status" in df.columns:
        df['is_furnished'] = df.status.eq("furnished").fillna(False).astype(int)
        df['is_semi_furnished'] = df.status.eq("semi-furnished").fillna(False).astype(int)
        df['is_unfurnished'] = df.status.eq("unfurnished").fillna(False).astype(int)


    if "verification_days" in df.columns:
        df['listing_age_days'] = df['verification_days']


    if "description" in df.columns:
        keywords = {
            "parking": "has_parking",
            "garden": "has_garden",
            "cctv": "has_cctv",
            "power backup": "has_power_backup",
            "hospital": "near_hospital",
            "metro": "near_metro",
        }

        for word, col in keywords.items():
            df[col] = df['description'].apply(
                lambda x: 1 if isinstance(x, str) and word in x else 0
            )


    if include_target_features and "price" in df.columns and "bkh" in df.columns:
        df['price_per_bhk'] = df.apply(
            lambda row: row['price'] / row['bhk'] if pd.notna(row['price']) and pd.notna(row['bhk']) and row['bhk'] > 0 else None, axis=1)


    if {"house_size", "bhk"}.issubset(df.columns):
        df['area_per_room'] = df.house_size / df.bhk.where(df.bhk > 0)
    if {"num_bathrooms", "bhk"}.issubset(df.columns):
        df['bathrooms_per_room'] = df.num_bathrooms / df.bhk.where(df.bhk > 0)

    return df
        


if __name__ == "__main__":
    from src.preprocessing import preprocess_house_data
    from src.data_loader import load_csvs_from_gcs

    # load the data and preprocess
    df = load_csvs_from_gcs("gs://riva_ml_datasets/house_rent/*.csv")
    df = preprocess_house_data(df)
    

    print(df.head())



