"""
FilterManager jerárquico para gestionar filtros en cascada.

Maneja la jerarquía: Equipo → Tipo → Modo de Falla
Cuando cambia un nivel superior, se resetean automáticamente los niveles inferiores.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from src.reliability_analysis.utils.logger_config import setup_logging

logger = setup_logging("FilterManager")


@dataclass
class FilterState:
    """
    Estado interno de los filtros.
    
    Attributes:
        equipment: Equipos seleccionados
        types: Tipos seleccionados
        failure_modes: Modos de falla seleccionados
    """
    equipment: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


class FilterManager:
    """
    Gestor de filtros jerárquico para datos de confiabilidad.
    
    Maneja la cascada de filtros: cuando cambia un nivel superior,
    los inferiores se resetean automáticamente.
    
    Attributes:
        data: DataFrame con los datos completos
        state: Estado actual de los filtros
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el FilterManager.
        
        Args:
            data: DataFrame completo con columnas 'Equipo', 'Tipo', 'mdf'
            
        Raises:
            ValueError: Si el DataFrame no tiene las columnas requeridas
        """
        required_cols = {'Equipo', 'Tipo', 'mdf'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"DataFrame debe contener las columnas: {required_cols}")
        
        self.data = data.copy()
        self.state = FilterState()
        logger.info("FilterManager inicializado")
        logger.debug(f"Datos: {len(self.data)} registros con columnas {list(self.data.columns)}")
    
    def set_equipment(self, equipment: List[str]) -> bool:
        """
        Establece los equipos seleccionados y resetea tipos y modos.
        
        Valida que los equipos existan en los datos antes de asignarlos.
        Resetea automáticamente tipos y modos de falla.
        
        Args:
            equipment: Lista de equipos a filtrar
            
        Returns:
            bool: True si la operación fue exitosa, False si hubo error de validación
        """
        if not isinstance(equipment, list):
            logger.error(f"equipment debe ser una lista, recibido: {type(equipment)}")
            return False
        
        available_equipment = self.data['Equipo'].unique().tolist()
        invalid_equipment = [e for e in equipment if e not in available_equipment]
        
        if invalid_equipment:
            logger.warning(
                f"Equipos inválidos: {invalid_equipment}. "
                f"Equipos disponibles: {available_equipment}"
            )
            return False
        
        # Cambiar equipos resetea tipos y modos
        self.state.equipment = equipment
        self.state.types = []
        self.state.failure_modes = []
        
        logger.info(f"Equipos establecidos: {equipment}")
        return True
    
    def set_types(self, types: List[str]) -> bool:
        """
        Establece los tipos seleccionados y resetea modos de falla.
        
        Valida que los tipos existan para los equipos seleccionados.
        Resetea automáticamente los modos de falla.
        
        Args:
            types: Lista de tipos a filtrar
            
        Returns:
            bool: True si la operación fue exitosa, False si hubo error de validación
        """
        if not isinstance(types, list):
            logger.error(f"types debe ser una lista, recibido: {type(types)}")
            return False
        
        # Obtener tipos válidos para equipos actuales
        valid_types = self.get_types_for_equipment()
        invalid_types = [t for t in types if t not in valid_types]
        
        if invalid_types:
            logger.warning(
                f"Tipos inválidos: {invalid_types}. "
                f"Tipos disponibles para equipos {self.state.equipment}: {valid_types}"
            )
            return False
        
        # Cambiar tipos resetea modos
        self.state.types = types
        self.state.failure_modes = []
        
        logger.info(f"Tipos establecidos: {types}")
        return True
    
    def set_failure_modes(self, modes: List[str]) -> bool:
        """
        Establece los modos de falla seleccionados.
        
        Valida que los modos de falla existan para los tipos seleccionados.
        
        Args:
            modes: Lista de modos de falla a filtrar
            
        Returns:
            bool: True si la operación fue exitosa, False si hubo error de validación
        """
        if not isinstance(modes, list):
            logger.error(f"modes debe ser una lista, recibido: {type(modes)}")
            return False
        
        valid_modes = self.get_failure_modes_for_types()
        invalid_modes = [m for m in modes if m not in valid_modes]
        
        if invalid_modes:
            logger.warning(
                f"Modos inválidos: {invalid_modes}. "
                f"Modos disponibles para tipos {self.state.types}: {valid_modes}"
            )
            return False
        
        self.state.failure_modes = modes
        logger.info(f"Modos de falla establecidos: {modes}")
        return True
    
    def get_types_for_equipment(self) -> List[str]:
        """
        Obtiene todos los tipos disponibles para equipos seleccionados.
        
        Si no hay equipos seleccionados, retorna todos los tipos disponibles.
        
        Returns:
            Lista de tipos únicos disponibles
        """
        if not self.state.equipment:
            # Sin filtro de equipo, retorna todos
            types = self.data['Tipo'].unique().tolist()
        else:
            df_filtered = self.data[self.data['Equipo'].isin(self.state.equipment)]
            types = df_filtered['Tipo'].unique().tolist()
        
        logger.debug(f"Tipos disponibles para {self.state.equipment}: {types}")
        return types
    
    def get_failure_modes_for_types(self) -> List[str]:
        """
        Obtiene todos los modos de falla disponibles para tipos seleccionados.
        
        Respeta tanto el filtro de equipos como el de tipos.
        Si no hay tipos seleccionados, retorna todos los modos para equipos.
        
        Returns:
            Lista de modos de falla únicos disponibles
        """
        df_filtered = self.data.copy()
        
        # Aplicar filtro de equipos
        if self.state.equipment:
            df_filtered = df_filtered[df_filtered['Equipo'].isin(self.state.equipment)]
        
        # Aplicar filtro de tipos
        if self.state.types:
            df_filtered = df_filtered[df_filtered['Tipo'].isin(self.state.types)]
        
        modes = df_filtered['mdf'].unique().tolist()
        logger.debug(
            f"Modos disponibles para equipo={self.state.equipment}, "
            f"tipos={self.state.types}: {modes}"
        )
        return modes
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Retorna el DataFrame filtrado según el estado actual.
        
        Aplica los filtros de equipos, tipos y modos de falla en cascada.
        
        Returns:
            DataFrame filtrado. Retorna DataFrame vacío si no hay filtros o
            si los filtros no coinciden con datos
        """
        df_filtered = self.data.copy()
        
        # Aplicar filtro de equipos
        if self.state.equipment:
            df_filtered = df_filtered[df_filtered['Equipo'].isin(self.state.equipment)]
        
        # Aplicar filtro de tipos
        if self.state.types:
            df_filtered = df_filtered[df_filtered['Tipo'].isin(self.state.types)]
        
        # Aplicar filtro de modos de falla
        if self.state.failure_modes:
            df_filtered = df_filtered[df_filtered['mdf'].isin(self.state.failure_modes)]
        
        logger.info(
            f"Datos filtrados: {len(df_filtered)} registros "
            f"(equipo={self.state.equipment}, "
            f"tipos={self.state.types}, "
            f"modos={self.state.failure_modes})"
        )
        return df_filtered
    
    def get_state(self) -> Dict[str, Any]:
        """
        Retorna el estado actual de los filtros.
        
        Returns:
            Diccionario con el estado actual de equipos, tipos y modos
        """
        state_dict = {
            'equipment': self.state.equipment.copy(),
            'types': self.state.types.copy(),
            'failure_modes': self.state.failure_modes.copy(),
            'filtered_count': len(self.get_filtered_data()),
            'total_count': len(self.data)
        }
        logger.debug(f"Estado actual: {state_dict}")
        return state_dict
    
    def reset(self) -> None:
        """
        Resetea todos los filtros a su estado inicial.
        
        Limpia los equipos, tipos y modos de falla seleccionados.
        """
        self.state = FilterState()
        logger.info("Filtros reseteados")
    
    def get_available_equipment(self) -> List[str]:
        """
        Obtiene todos los equipos disponibles en los datos.
        
        Returns:
            Lista de equipos únicos disponibles
        """
        equipment = self.data['Equipo'].unique().tolist()
        logger.debug(f"Equipos disponibles: {equipment}")
        return equipment
    
    def get_available_types(self) -> List[str]:
        """
        Obtiene todos los tipos disponibles (sin filtros).
        
        Returns:
            Lista de tipos únicos disponibles
        """
        types = self.data['Tipo'].unique().tolist()
        logger.debug(f"Tipos disponibles: {types}")
        return types
    
    def get_available_failure_modes(self) -> List[str]:
        """
        Obtiene todos los modos de falla disponibles (sin filtros).
        
        Returns:
            Lista de modos de falla únicos disponibles
        """
        modes = self.data['mdf'].unique().tolist()
        logger.debug(f"Modos disponibles: {modes}")
        return modes
