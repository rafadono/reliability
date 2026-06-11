"""
Hierarchical filter manager.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from src.reliability_analysis.utils.logger_config import setup_logging

logger = setup_logging("FilterManager")


@dataclass
class FilterState:
    equipment: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


class FilterManager:
    """
    Manages cascade filtering: Equipment -> Type -> Failure Mode.
    """

    def __init__(self, data: pd.DataFrame):
        required_cols = {"Equipment", "Type", "mdf"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.data = data.copy()
        self.state = FilterState()
        logger.info("FilterManager initialized")

    def set_equipment(self, equipment: List[str]) -> bool:
        if not isinstance(equipment, list):
            logger.error("equipment must be a list")
            return False

        available = self.data["Equipment"].unique().tolist()
        invalid = [e for e in equipment if e not in available]

        if invalid:
            logger.warning(f"Invalid equipment: {invalid}")
            return False

        self.state.equipment = equipment
        self.state.types = []
        self.state.failure_modes = []
        return True

    def set_types(self, types: List[str]) -> bool:
        if not isinstance(types, list):
            logger.error("types must be a list")
            return False

        valid_types = self.get_types_for_equipment()
        invalid = [t for t in types if t not in valid_types]

        if invalid:
            logger.warning(f"Invalid types: {invalid}")
            return False

        self.state.types = types
        self.state.failure_modes = []
        return True

    def set_failure_modes(self, modes: List[str]) -> bool:
        if not isinstance(modes, list):
            logger.error("modes must be a list")
            return False

        valid_modes = self.get_failure_modes_for_types()
        invalid = [m for m in modes if m not in valid_modes]

        if invalid:
            logger.warning(f"Invalid modes: {invalid}")
            return False

        self.state.failure_modes = modes
        return True

    def get_types_for_equipment(self) -> List[str]:
        if not self.state.equipment:
            return self.data["Type"].unique().tolist()
        df_filtered = self.data[self.data["Equipment"].isin(self.state.equipment)]
        return df_filtered["Type"].unique().tolist()

    def get_failure_modes_for_types(self) -> List[str]:
        df_filtered = self.data.copy()
        if self.state.equipment:
            df_filtered = df_filtered[
                df_filtered["Equipment"].isin(self.state.equipment)
            ]
        if self.state.types:
            df_filtered = df_filtered[df_filtered["Type"].isin(self.state.types)]
        return df_filtered["mdf"].unique().tolist()

    def get_filtered_data(self) -> pd.DataFrame:
        df_filtered = self.data.copy()
        if self.state.equipment:
            df_filtered = df_filtered[
                df_filtered["Equipment"].isin(self.state.equipment)
            ]
        if self.state.types:
            df_filtered = df_filtered[df_filtered["Type"].isin(self.state.types)]
        if self.state.failure_modes:
            df_filtered = df_filtered[df_filtered["mdf"].isin(self.state.failure_modes)]
        return df_filtered

    def get_state(self) -> Dict[str, Any]:
        return {
            "equipment": self.state.equipment.copy(),
            "types": self.state.types.copy(),
            "failure_modes": self.state.failure_modes.copy(),
            "filtered_count": len(self.get_filtered_data()),
            "total_count": len(self.data),
        }

    def reset(self) -> None:
        self.state = FilterState()
        logger.info("Filters reset")

    def get_available_equipment(self) -> List[str]:
        return self.data["Equipment"].unique().tolist()

    def get_available_types(self) -> List[str]:
        return self.data["Type"].unique().tolist()

    def get_available_failure_modes(self) -> List[str]:
        return self.data["mdf"].unique().tolist()
