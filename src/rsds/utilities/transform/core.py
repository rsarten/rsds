import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def select_numeric(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes(include=np.number)  # type:ignore


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    numeric_data = select_numeric(data)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)
    numeric_scaled = pd.DataFrame(numeric_scaled)
    numeric_scaled.columns = numeric_data.columns
    return numeric_scaled
