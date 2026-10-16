from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from fnmatch import fnmatch

import pandas as pd
from google.cloud import storage

# parse the gcs url
def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Convert a GCS URI into bucket name and object path.
    Example: gs://riva_ml_datasets/house_rent/*.csv
    Return:
        bucket_name = riva_ml_datasets
        object_path = house_rent/*.csv

    """
    parsed_uri = urlparse(gcs_uri)
    if parsed_uri.scheme != "gs":
        raise ValueError(
            f"Expected a gs:// path, but received {gcs_uri}"
        )

    bucket_name = parsed_uri.netloc
    object_path = parsed_uri.path.lstrip("/")
    # ipdb.set_trace()

    return bucket_name, object_path




def load_csvs_from_gcs(gcs_uri: str, project_id: str) -> pd.DataFrame:
    """
    Load and combine all the CSV files from GCS bucket
    """
    bucket_name, object_pattern = parse_gcs_uri(gcs_uri)
    # Listing by the static prefix is efficient; fnmatch then enforces the
    # complete pattern (for example, it will not ingest JSON or nested CSVs
    # that happen to share the same prefix).
    wildcard_positions = [position for position in (object_pattern.find("*"), object_pattern.find("?")) if position >= 0]
    prefix = object_pattern[:min(wildcard_positions)] if wildcard_positions else object_pattern

    storage_client = storage.Client(project=project_id)

    blobs = storage_client.list_blobs(bucket_or_name=bucket_name, prefix=prefix)
    csv_blobs = [
        blob for blob in blobs
        if blob.name.lower().endswith(".csv") and fnmatch(blob.name, object_pattern)
    ]

    if not csv_blobs:
        raise FileNotFoundError(
            f"No CSV files were found at {gcs_uri}"
        )

    dataframes: list[pd.DataFrame] = []
    # ipdb.set_trace()


    for blob in sorted(csv_blobs, key=lambda i: i.name):
        file_bytes = blob.download_as_bytes()

        dataframe = pd.read_csv(
            BytesIO(file_bytes),
            low_memory=False
        )

        dataframe["source_file"] = Path(blob.name).name
        dataframes.append(dataframe)
        # ipdb.set_trace()

    combined_dataframe = pd.concat(dataframes, ignore_index=True, sort=False)

    print(f"GCS URI: {gcs_uri}")
    print(f"Files loaded: {len(csv_blobs)}")
    print(f"Combined rows: {len(combined_dataframe)}")


    return combined_dataframe


def load_local_csvs(directory_path: str | Path,) -> pd.DataFrame:
    """
    Optional function for testing with locally downloaded csv files.
    """
    directory_path = Path(directory_path)
    csvs_path = sorted(directory_path.glob("*.csv"))

    if not csvs_path:
        raise FileNotFoundError(
            f"No CSV files were found at {directory_path}"
        )

    dataframes: list[pd.DataFrame] = []

    for csv_path in csvs_path:
        dataframe = pd.read_csv(csv_path, low_memory=False)

        dataframe["source_file"] = csv_path.name

        dataframes.append(dataframe)
    return pd.concat(dataframes, ignore_index=True, sort=False)




# print(load_csvs_from_gcs("gs://riva_ml_datasets/house_rent/*.csv", "ai-ml-projects-260916"))



    





