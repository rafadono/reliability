"""
Pareto analysis module for reliability data.

Analyzes failure distribution respecting hierarchical filters:
Equipment → Type → Failure Mode.

Provides Pareto charts showing the 80/20 rule distribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class ParetoAnalyzer:
    """
    Analyzes failure data using Pareto principle.

    Respects hierarchical structure: filters applied at each level
    ensure data integrity and meaningful analysis.
    """

    @staticmethod
    def analyze_by_equipment(
        df: pd.DataFrame, failure_column: str = "Equipment"
    ) -> Dict[str, any]:
        """
        Pareto analysis by equipment.

        Args:
            df: Filtered dataframe
            failure_column: Column name for equipment

        Returns:
            {
                'items': [...equipment names...],
                'counts': [...failure counts...],
                'cumsum_pct': [...cumulative %...],
                'total': int
            }
        """
        value_counts = df[failure_column].value_counts().sort_values(ascending=False)

        total = value_counts.sum()
        cumsum = value_counts.cumsum()
        cumsum_pct = (cumsum / total * 100).round(2)

        return {
            "items": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "cumsum_pct": cumsum_pct.tolist(),
            "total": int(total),
            "items_for_80pct": int(np.searchsorted(cumsum_pct.values, 80)) + 1,
        }

    @staticmethod
    def analyze_by_type(
        df: pd.DataFrame,
        equipment: Optional[str] = None,
        type_column: str = "Type",
        equipment_column: str = "Equipment",
    ) -> Dict[str, any]:
        """
        Pareto analysis by type (respects equipment filter).

        Args:
            df: Dataframe
            equipment: Filter by specific equipment (optional)
            type_column: Column name for type
            equipment_column: Column name for equipment

        Returns:
            Pareto dict with items filtered by equipment
        """
        if equipment:
            df = df[df[equipment_column] == equipment]

        if df.empty:
            return {"items": [], "counts": [], "cumsum_pct": [], "total": 0}

        value_counts = df[type_column].value_counts().sort_values(ascending=False)

        total = value_counts.sum()
        cumsum = value_counts.cumsum()
        cumsum_pct = (cumsum / total * 100).round(2)

        return {
            "items": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "cumsum_pct": cumsum_pct.tolist(),
            "total": int(total),
            "items_for_80pct": int(np.searchsorted(cumsum_pct.values, 80)) + 1,
        }

    @staticmethod
    def analyze_by_failure_mode(
        df: pd.DataFrame,
        equipment: Optional[str] = None,
        failure_type: Optional[str] = None,
        mode_column: str = "mdf",
        equipment_column: str = "Equipment",
        type_column: str = "Type",
    ) -> Dict[str, any]:
        """
        Pareto analysis by failure mode (respects equipment + type filters).

        Hierarchical: filters cascade down from equipment → type → mode.

        Args:
            df: Dataframe
            equipment: Filter by equipment
            failure_type: Filter by type
            mode_column: Column name for failure mode
            equipment_column: Column name for equipment
            type_column: Column name for type

        Returns:
            Pareto dict with full hierarchy respected
        """
        if equipment:
            df = df[df[equipment_column] == equipment]

        if failure_type:
            df = df[df[type_column] == failure_type]

        if df.empty:
            return {"items": [], "counts": [], "cumsum_pct": [], "total": 0}

        value_counts = df[mode_column].value_counts().sort_values(ascending=False)

        total = value_counts.sum()
        cumsum = value_counts.cumsum()
        cumsum_pct = (cumsum / total * 100).round(2)

        return {
            "items": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "cumsum_pct": cumsum_pct.tolist(),
            "total": int(total),
            "items_for_80pct": int(np.searchsorted(cumsum_pct.values, 80)) + 1,
        }

    @staticmethod
    def get_80_20_split(
        analysis: Dict[str, any],
    ) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        Split Pareto data into "vital few" (80%) and "trivial many" (20%).

        Args:
            analysis: Result from analyze_* methods

        Returns:
            (vital_items, trivial_items, stats)
        """
        if not analysis["items"]:
            return [], [], {"vital": 0, "trivial": 0}

        split_index = analysis["items_for_80pct"]
        vital = analysis["items"][:split_index]
        trivial = analysis["items"][split_index:]

        return (
            vital,
            trivial,
            {
                "vital_count": len(vital),
                "trivial_count": len(trivial),
                "vital_percentage": float(analysis["cumsum_pct"][split_index - 1])
                if split_index > 0
                else 0,
            },
        )


# Module level functions for test compatibility (returning list of dicts)
def analyze_by_equipment(df, failure_column="Equipment"):
    value_counts = df[failure_column].value_counts().sort_values(ascending=False)
    total = value_counts.sum()
    if total == 0:
        return []
    cumsum = value_counts.cumsum()
    cumsum_pct = (cumsum / total * 100).round(2)
    return [
        {"equipment": name, "failures": int(count), "cumulative_percentage": float(pct)}
        for name, count, pct in zip(
            value_counts.index, value_counts.values, cumsum_pct.values
        )
    ]


def analyze_by_type(
    df, equipment=None, type_column="Type", equipment_column="Equipment"
):
    if equipment:
        df = df[df[equipment_column] == equipment]
    if df.empty:
        return []
    value_counts = df[type_column].value_counts().sort_values(ascending=False)
    total = value_counts.sum()
    if total == 0:
        return []
    cumsum = value_counts.cumsum()
    cumsum_pct = (cumsum / total * 100).round(2)
    return [
        {"type": name, "failures": int(count), "cumulative_percentage": float(pct)}
        for name, count, pct in zip(
            value_counts.index, value_counts.values, cumsum_pct.values
        )
    ]


def analyze_by_failure_mode(
    df,
    equipment=None,
    failure_type=None,
    mode_column="mdf",
    equipment_column="Equipment",
    type_column="Type",
):
    if equipment:
        df = df[df[equipment_column] == equipment]
    if failure_type:
        df = df[df[type_column] == failure_type]
    if df.empty:
        return []
    # If the dataframe has 'Modo de Falla' instead of 'mdf', use it
    col = mode_column
    if col not in df.columns and "Modo de Falla" in df.columns:
        col = "Modo de Falla"
    if col not in df.columns:
        # Fallback to column case-insensitive matching
        matched_cols = [c for c in df.columns if c.lower() == col.lower()]
        if matched_cols:
            col = matched_cols[0]
        else:
            return []
    value_counts = df[col].value_counts().sort_values(ascending=False)
    total = value_counts.sum()
    if total == 0:
        return []
    cumsum = value_counts.cumsum()
    cumsum_pct = (cumsum / total * 100).round(2)
    return [
        {"mode": name, "count": int(count), "cumulative_percentage": float(pct)}
        for name, count, pct in zip(
            value_counts.index, value_counts.values, cumsum_pct.values
        )
    ]


def get_80_20_split(result):
    if not result:
        return {"items_for_80_percent": 0, "cumulative_percentage": 0.0}
    for i, item in enumerate(result):
        if item.get("cumulative_percentage", 0.0) >= 80.0:
            return {
                "items_for_80_percent": i + 1,
                "cumulative_percentage": float(item.get("cumulative_percentage")),
            }
    return {
        "items_for_80_percent": len(result),
        "cumulative_percentage": float(result[-1].get("cumulative_percentage", 100.0)),
    }
