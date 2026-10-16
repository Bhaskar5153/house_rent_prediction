import pandas as pd
import numpy as np
import re
import ipdb


from src.data_loader import load_csvs_from_gcs
from src.prepare_features import feature_engineering

df = load_csvs_from_gcs("gs://riva_ml_datasets/house_rent/*.csv", "ai-ml-projects-260916")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    COLUMN_ALIAS = {
        "numbathrooms": "num_bathrooms",
        "numbalconies": "num_balconies",
        "isnegotiable": "is_negotiable",
        "pricesqft": "price_sqft",
        "verificationdate": "verification_date",
        "securitydeposit": "security_deposite",
    }

    renamed_columns = {}
    for column in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
        if key in COLUMN_ALIAS:
            renamed_columns[column] = COLUMN_ALIAS[key]
        else:
            renamed_columns[column] = column

    df = df.rename(columns=renamed_columns)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # ipdb.set_trace()
    return df


# print(normalize_columns(df=df))

def clean_description(text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return None

    text = str(text)

    text = text.replace("Ã¢Â€Â™", "'").replace("Ã¢Â€Â", "")
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(pattern=r"[^\w\s]", repl=" ", string=text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    # ipdb.set_trace()
    return text

# print(clean_description("ItÃ¢Â€Â™s a 3 bhk independent house situated in Model Town, Delhi. It has area of 1020 sqft and is available at a rent of Rs. 48,000. It has 1 tv, 1 refrigerator, 1 sofa and 1 ac. It also has 1 bed. This residential property is ready-to-move-in. Please contact for more details. "))



def preprocess_house_data(df: pd.DataFrame, training: bool = True, model_ready: bool = True) -> pd.DataFrame:
    df = normalize_columns(df)

    # location and city
    if "location" in df.columns:
        df['location'] = df['location'].astype(str).str.strip().str.lower()

    if "city" in df.columns:
        df["city"] = df['city'].astype(str).str.strip().str.lower()

    if "city" in df.columns:
        df = df[df['city'] != 'hisar'].copy()

    # house type
    if "house_type" in df.columns:
        df["bhk"] = df["house_type"].str.extract(r"(\d+)\s*bhk", flags=re.IGNORECASE).fillna(0).astype(int)
        df["property_type"] = df['house_type'].str.extract(
            r"(apartment|independent floor|independent house)", flags=re.IGNORECASE
        )

    # house size
    if "house_size" in df.columns:
        df['house_size'] = (df["house_size"].astype(str).str.replace("sq ft", "", regex=False).str.replace(",", "", regex=False).str.strip().replace("", None))
        df["house_size"] = pd.to_numeric(df["house_size"], errors="coerce")



    # latitude and longitude
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # price
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # currency
    if "currency" in df.columns:
        df['currency'] = df['currency'].astype(str).str.strip().str.upper()

        if df['currency'].nunique() == 1:
            df = df.drop(columns=["currency"])

    for col in ['num_bathrooms', 'num_balconies']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

        

    if "verification_date" in df.columns:
        df["verification_days"] = df['verification_date'].str.extract(r"(\d+)").astype(float)
        df["verification_days"] = df["verification_days"].fillna(
            df['verification_days'].map({
            "Posted a day ago": 1,
            "Posted a month ago": 31,
            "Posted 2 months ago": 60,
            "Posted 1 years ago": 365,
            "Posted 2 years ago": 730,
            "Posted 3 years ago": 1095})
        )


    if "Status" in df.columns:
        df['status'] = df['Status'].astype(str).str.lower()

    if "security_deposite" in df.columns:
        df['security_deposite'] = (
            df['security_deposite'].replace("No Deposite", 0).replace("null", None)
        )
        df['security_deposite'] = pd.to_numeric(df['security_deposite'], errors='coerce')


    if "description" in df.columns:
        df['description'] = df['description'].apply(func=clean_description)

    if "is_negotiable" in df.columns:
        df['is_negotiable'] = df['is_negotiable'].astype(str).str.lower()
        df['is_negotiable'] = df['is_negotiable'].map({"yes": 1, "true": 1, "no": 0, "false": 0})
        df['is_negotiable'] = df['is_negotiable'].fillna(0).astype(int)

    # ipdb.set_trace()
    for col in ["house_size", "bhk", "num_bathrooms", "num_balconies"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            invalid = df[col].lt(0) if col in ["num_bathrooms", "num_balconies"] else df[col].le(0)
            df.loc[invalid, col] = np.nan

    df = feature_engineering(df)
    if model_ready:
        df = select_model_dataframe(df, training=training)
    return df.reset_index(drop=True)


def select_model_dataframe(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    result = df.reindex(columns=df.columns).copy()

    for col in df.select_dtypes(exclude="object").columns:
        result[col] = pd.to_numeric(result[col], errors='coerce')
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(exclude=['int', 'float']).columns:
        value = result[col].astype("str").str.strip().str.lower()
        value = value.replace({'':pd.NA, 'null': pd.NA, 'nan':pd.NA, "none": pd.NA})
        result[col] = value.astype(object).where(value.notna(), np.nan)

    if training:
        if "price" not in df:
            raise ValueError("Training requires the price")

        result['price'] = pd.to_numeric(df.price, errors='coerce')
        result = result[result.price.gt(0) & np.isfinite(result.price) & result.city.isin(["delhi", "mumbai", "pune"])].drop_duplicates()
        # ipdb.set_trace()
    return result


    









        
    # return df

    

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.config import load_config
    from src.data_loader import load_local_csvs, load_csvs_from_gcs

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="Local csv or directory: defaults to configured GCS")
    parser.add_argument('--ouput', default="artifacts/processed.csv")
    args = parser.parse_args()

    if args.input:
        source = Path(args.input)
        df = load_local_csvs(source) if source.is_dir else pd.read_csv(source)

    else:
        config = load_config('configs/config.yml')

        df = load_csvs_from_gcs(config.data_uri, config.project_id)

    df = preprocess_house_data(df)
    
    output = Path(args.ouput)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    # ipdb.set_trace()
    print(df.head())
    


