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
    plant: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


class FilterManager:
    """
    Manages hierarchical cascade filtering: Plant -> Equipment -> Type -> Failure Mode.
    """

    def __init__(self, data: pd.DataFrame):
        required_cols = {"Equipment", "Type", "mdf"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.data = data.copy()
        self.state = FilterState()
        logger.info("FilterManager initialized")

    def set_plant(self, plant: List[str]) -> bool:
        if not isinstance(plant, list):
            logger.error("plant must be a list")
            return False

        if "Plant" in self.data.columns:
            available = self.data["Plant"].dropna().unique().tolist()
            valid = [p for p in plant if p in available]
            self.state.plant = valid
        else:
            self.state.plant = []

        self.state.equipment = []
        self.state.types = []
        self.state.failure_modes = []
        return True

    def set_equipment(self, equipment: List[str]) -> bool:
        if not isinstance(equipment, list):
            logger.error("equipment must be a list")
            return False

        valid_equipment = self.get_equipment_for_plant()
        valid = [e for e in equipment if e in valid_equipment]
        self.state.equipment = valid
        self.state.types = []
        self.state.failure_modes = []
        return True

    def set_types(self, types: List[str]) -> bool:
        if not isinstance(types, list):
            logger.error("types must be a list")
            return False

        valid_types = self.get_types_for_equipment()
        valid = [t for t in types if t in valid_types]
        self.state.types = valid
        self.state.failure_modes = []
        return True

    def set_failure_modes(self, modes: List[str]) -> bool:
        if not isinstance(modes, list):
            logger.error("modes must be a list")
            return False

        valid_modes = self.get_failure_modes_for_types()
        valid = [m for m in modes if m in valid_modes]
        self.state.failure_modes = valid
        return True

    def get_available_plants(self) -> List[str]:
        if "Plant" in self.data.columns:
            return sorted([str(x) for x in self.data["Plant"].dropna().unique().tolist()])
        return []

    def get_equipment_for_plant(self) -> List[str]:
        df_filtered = self.data.copy()
        if self.state.plant and "Plant" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Plant"].isin(self.state.plant)]
        return sorted([str(x) for x in df_filtered["Equipment"].dropna().unique().tolist()])

    def get_types_for_equipment(self) -> List[str]:
        df_filtered = self.data.copy()
        if self.state.plant and "Plant" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Plant"].isin(self.state.plant)]
        if self.state.equipment:
            df_filtered = df_filtered[df_filtered["Equipment"].isin(self.state.equipment)]
        return sorted([str(x) for x in df_filtered["Type"].dropna().unique().tolist()])

    def get_failure_modes_for_types(self) -> List[str]:
        df_filtered = self.data.copy()
        if self.state.plant and "Plant" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Plant"].isin(self.state.plant)]
        if self.state.equipment:
            df_filtered = df_filtered[df_filtered["Equipment"].isin(self.state.equipment)]
        if self.state.types:
            df_filtered = df_filtered[df_filtered["Type"].isin(self.state.types)]
        return sorted([str(x) for x in df_filtered["mdf"].dropna().unique().tolist()])

    def get_filtered_data(self) -> pd.DataFrame:
        df_filtered = self.data.copy()
        if self.state.plant and "Plant" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Plant"].isin(self.state.plant)]
        if self.state.equipment:
            df_filtered = df_filtered[df_filtered["Equipment"].isin(self.state.equipment)]
        if self.state.types:
            df_filtered = df_filtered[df_filtered["Type"].isin(self.state.types)]
        if self.state.failure_modes:
            df_filtered = df_filtered[df_filtered["mdf"].isin(self.state.failure_modes)]
        return df_filtered

    def get_state(self) -> Dict[str, Any]:
        return {
            "plant": self.state.plant.copy(),
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
        return self.get_equipment_for_plant()

    def get_available_types(self) -> List[str]:
        return self.get_types_for_equipment()

    def get_available_failure_modes(self) -> List[str]:
        return self.get_failure_modes_for_types()
