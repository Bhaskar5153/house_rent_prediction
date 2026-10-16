from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class Settings:
    project_id: str
    data_uri: str
    target: str
    test_size: float
    random_state: int
    outlier_iqr_multiplier: float
    fit_intercept: bool
    artifact_dir: Path



def load_config(config_path: str | Path) -> Settings:
    with open(file=config_path, mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return Settings(
        project_id=str(config["gcp"]["project_id"]),
        data_uri=str(config["gcp"]["data_uri"]),
        target=str(config["data"]["target"]),
        test_size=float(config["data"].get("test_size", 0.20)),
        random_state=int(config["data"].get("random_state", 42)),
        outlier_iqr_multiplier=float(config["data"].get("outlier_iqr_multiplier", 1.5)),
        fit_intercept=bool(config["model"].get("fit_intercept", True)),
        artifact_dir=Path(config["output"].get("artifact_dir", "artifacts"))

    )

    





