"""
Clase para el procesamiento de datos de confiabilidad.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Procesador de datos para limpiar, formatear y estructurar
    datasets de análisis de confiabilidad.
    """
    
    @staticmethod
    def treat_data(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Renombrar columnas para estandarizar (case-insensitive)
        rename_dict = {}
        for col in result.columns:
            if col.lower() in ['modo de falla', 'mdf']:
                rename_dict[col] = 'mdf'
            elif col.lower() in ['duracion', 'ttx']:
                rename_dict[col] = 'TTX'
        result = result.rename(columns=rename_dict)
        
        # Parsear fechas si existen
        if 'Fecha' in result.columns and 'Hora' in result.columns:
            result['Fecha_Inicio'] = pd.to_datetime(result['Fecha'] + ' ' + result['Hora'], dayfirst=True, format='mixed', errors='coerce')
        elif 'Fecha' in result.columns:
            result['Fecha_Inicio'] = pd.to_datetime(result['Fecha'], dayfirst=True, format='mixed', errors='coerce')
        
        # Garantizar que exista la columna TTX (tiempo de reparacion / downtime)
        if 'TTX' not in result.columns:
            if 'Dias' in result.columns:
                result['TTX'] = result['Dias']
            else:
                result['TTX'] = 0.0

        # Remover duplicados
        result = result.drop_duplicates()
        
        # Asegurar que las columnas numéricas sean float (manejando comas si existen)
        for col in ['TTX', 'Dias', 'TBX']:
            if col in result.columns:
                if result[col].dtype == 'O':
                    result[col] = result[col].astype(str).str.replace(',', '.')
                result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0)

        # Calcular Fecha_Fin y auto-calcular TBX (Tiempo Entre Fallas)
        if 'Fecha_Inicio' in result.columns:
            if 'TTX' in result.columns:
                result['Fecha_Fin'] = result['Fecha_Inicio'] + pd.to_timedelta(result['TTX'], unit='h')
            else:
                result['Fecha_Fin'] = result['Fecha_Inicio']
                
            # Si no existe TBX, lo calculamos como la diferencia entre fallas
            if 'TBX' not in result.columns and 'Equipo' in result.columns:
                result = result.sort_values(['Equipo', 'Fecha_Inicio'])
                result['Prev_Fecha_Fin'] = result.groupby('Equipo')['Fecha_Fin'].shift(1)
                result['TBX'] = (result['Fecha_Inicio'] - result['Prev_Fecha_Fin']).dt.total_seconds() / 3600.0
                result['TBX'] = result['TBX'].fillna(0.0).clip(lower=0.0)
                result = result.drop(columns=['Prev_Fecha_Fin'])
            
        return result

    @staticmethod
    def get_equipment_types(df: pd.DataFrame, equipo: str) -> list:
        if 'Equipo' not in df.columns or 'Tipo' not in df.columns:
            return []
        filtered = df[df['Equipo'] == equipo]
        return filtered['Tipo'].dropna().unique().tolist()
        
    @staticmethod
    def calculate_tbf(df: pd.DataFrame, equipo_nombre: str, tipos: list) -> pd.DataFrame:
        mask = (df['Equipo'] == equipo_nombre) & (df['Tipo'].isin(tipos))
        filtered = df[mask].copy()
        
        if filtered.empty or 'Fecha_Inicio' not in filtered.columns or 'Fecha_Fin' not in filtered.columns:
            filtered['TBX'] = 0.0
            return filtered
            
        filtered = filtered.sort_values('Fecha_Inicio').reset_index(drop=True)
        filtered['TBX'] = 0.0
        
        for i in range(1, len(filtered)):
            delta = filtered.loc[i, 'Fecha_Inicio'] - filtered.loc[i-1, 'Fecha_Fin']
            # TBX en horas
            filtered.loc[i, 'TBX'] = max(0.0, delta.total_seconds() / 3600.0)
            
        return filtered