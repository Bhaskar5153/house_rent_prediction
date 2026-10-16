import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
import ipdb


def remove_outlier_iqr(df: pd.DataFrame, cols: list, multiplier: float = 1.5, group_by: str | None = None) -> pd.DataFrame:
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("IQR multiplier must be a positive number")
    keep = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        if group_by is None:
            q1, q3 = values.quantile(0.25), values.quantile(.75)
        else:
            grouped = values.groupby(df[group_by], dropna=False)
            q1 = grouped.transform(lambda group: group.quantile(.25))
            q3 = grouped.transform(lambda group: group.quantile(.75))
        iqr = q3 - q1
        valid = values.between(q1 - multiplier * iqr, q3 + multiplier * iqr)
        keep &= (values.isna() | valid | (iqr<=0) | pd.isna(iqr).fillna(True))
    return df.loc[keep].copy()



def remove_outliers_zscore(df: pd.DataFrame, cols: list, z_thresh: float = 3.0,) -> pd.DataFrame:
    if not np.isfinite(z_thresh) or z_thresh <= 0:
        raise ValueError("z_threshould must finite and a positive number")

    keep = pd.Series(True, index=df.index)
    for col in cols:
        if col in df:
            values = pd.to_numeric(df[col], errors='coerce')
            std = values.std(ddof=0)
            if pd.notna(std) and std > 0:
                keep &= values.isna() | ((values - values.mean()).abs / std <= z_thresh)
                ipdb.set_trace()

    return df.loc[keep].copy()









