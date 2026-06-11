"""
Reliability data processor.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Cleans and standardizes reliability datasets.
    """

    @staticmethod
    def treat_data(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        rename_dict = {}
        for col in result.columns:
            col_lower = col.lower()
            if col_lower in ["equipo", "equipment"]:
                rename_dict[col] = "Equipment"
            elif col_lower in ["tipo", "type"]:
                rename_dict[col] = "Type"
            elif col_lower in ["modo de falla", "mdf", "failure mode"]:
                rename_dict[col] = "mdf"
            elif col_lower in ["duracion", "duración", "duration", "ttx", "downtime"]:
                rename_dict[col] = "TTX"
            elif col_lower in ["fecha", "date"]:
                rename_dict[col] = "Date"
            elif col_lower in ["hora", "time"]:
                rename_dict[col] = "Time"
            elif col_lower in ["dias", "días", "days"]:
                rename_dict[col] = "Days"
            elif col_lower in ["censurado", "censored"]:
                rename_dict[col] = "Censored"
            elif col_lower in ["comentario", "comment"]:
                rename_dict[col] = "Comment"

        result = result.rename(columns=rename_dict)

        if "Date" in result.columns and "Time" in result.columns:
            result["Start_Date"] = pd.to_datetime(
                result["Date"].astype(str) + " " + result["Time"].astype(str),
                dayfirst=True,
                format="mixed",
                errors="coerce",
            )
        elif "Date" in result.columns:
            result["Start_Date"] = pd.to_datetime(
                result["Date"], dayfirst=True, format="mixed", errors="coerce"
            )

        if "TTX" not in result.columns:
            if "Days" in result.columns:
                result["TTX"] = result["Days"]
            else:
                result["TTX"] = 0.0

        result = result.drop_duplicates()

        for col in ["TTX", "Days", "TBX"]:
            if col in result.columns:
                if result[col].dtype == "O":
                    result[col] = result[col].astype(str).str.replace(",", ".")
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

        if "Start_Date" in result.columns:
            if "TTX" in result.columns:
                result["End_Date"] = result["Start_Date"] + pd.to_timedelta(
                    result["TTX"], unit="h"
                )
            else:
                result["End_Date"] = result["Start_Date"]

            if "TBX" not in result.columns and "Equipment" in result.columns:
                result = result.sort_values(["Equipment", "Start_Date"])
                result["Prev_End_Date"] = result.groupby("Equipment")["End_Date"].shift(
                    1
                )
                result["TBX"] = (
                    result["Start_Date"] - result["Prev_End_Date"]
                ).dt.total_seconds() / 3600.0
                result["TBX"] = result["TBX"].fillna(0.0).clip(lower=0.0)
                result = result.drop(columns=["Prev_End_Date"])

        return result

    @staticmethod
    def get_equipment_types(df: pd.DataFrame, equipment: str) -> list:
        if "Equipment" not in df.columns or "Type" not in df.columns:
            return []
        filtered = df[df["Equipment"] == equipment]
        return filtered["Type"].dropna().unique().tolist()

    @staticmethod
    def calculate_tbf(
        df: pd.DataFrame, equipment_name: str, types: list
    ) -> pd.DataFrame:
        mask = (df["Equipment"] == equipment_name) & (df["Type"].isin(types))
        filtered = df[mask].copy()

        if (
            filtered.empty
            or "Start_Date" not in filtered.columns
            or "End_Date" not in filtered.columns
        ):
            filtered["TBX"] = 0.0
            return filtered

        filtered = filtered.sort_values("Start_Date").reset_index(drop=True)
        filtered["TBX"] = 0.0

        for i in range(1, len(filtered)):
            delta = filtered.loc[i, "Start_Date"] - filtered.loc[i - 1, "End_Date"]
            filtered.loc[i, "TBX"] = max(0.0, delta.total_seconds() / 3600.0)

        return filtered
