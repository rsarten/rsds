import pandas as pd


def col_present(target: str, data: pd.DataFrame) -> bool:
    return target in data.columns


def check_col_present(target: str, data: pd.DataFrame) -> None:
    if not col_present(target, data):
        raise LookupError(f"{target} is not member of feature set")
