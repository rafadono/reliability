"""
Pruebas unitarias para DataProcessor.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.reliability_analysis.core.data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Crea datos de ejemplo para pruebas."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    data = {
        'Fecha': [d.strftime('%d/%m/%Y') for d in dates],
        'Hora': ['08:00:00'] * 5,
        'Duracion': [1.0, 2.0, 1.5, 2.5, 1.0],
        'Tipo': ['MC', 'MC', 'MCE', 'MC', 'MP'],
        'Equipo': ['EQUIPO_A', 'EQUIPO_A', 'EQUIPO_A', 'EQUIPO_A', 'EQUIPO_A'],
        'Modo de Falla': ['Falla_1', 'Falla_2', 'Falla_1', 'Falla_3', 'Preventiva']
    }
    return pd.DataFrame(data)


class TestDataProcessor:
    
    def test_treat_data_creates_required_columns(self, sample_data):
        """Verifica que treat_data crea las columnas necesarias."""
        result = DataProcessor.treat_data(sample_data)
        
        required_cols = ['Fecha_Inicio', 'Fecha_Fin', 'TTX', 'Tipo', 'Equipo', 'mdf']
        assert all(col in result.columns for col in required_cols)
    
    def test_treat_data_date_parsing(self, sample_data):
        """Verifica que las fechas se parsean correctamente."""
        result = DataProcessor.treat_data(sample_data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Fecha_Inicio'])
        assert pd.api.types.is_datetime64_any_dtype(result['Fecha_Fin'])
        assert result['Fecha_Inicio'].iloc[0].year == 2024
    
    def test_treat_data_duration_calculation(self, sample_data):
        """Verifica que Fecha_Fin se calcula correctamente."""
        result = DataProcessor.treat_data(sample_data)
        
        for idx in result.index:
            expected_fin = result.loc[idx, 'Fecha_Inicio'] + timedelta(hours=result.loc[idx, 'TTX'])
            assert result.loc[idx, 'Fecha_Fin'] == expected_fin
    
    def test_treat_data_removes_duplicates(self):
        """Verifica que remove_duplicates funciona."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        data = {
            'Fecha': [d.strftime('%d/%m/%Y') for d in dates] + [dates[0].strftime('%d/%m/%Y')],
            'Hora': ['08:00:00'] * 4,
            'Duracion': [1.0, 2.0, 1.5, 1.0],
            'Tipo': ['MC', 'MC', 'MCE', 'MC'],
            'Equipo': ['EQUIPO_A'] * 4,
            'Modo de Falla': ['Falla_1', 'Falla_2', 'Falla_1', 'Falla_1']
        }
        df = pd.DataFrame(data)
        result = DataProcessor.treat_data(df)
        
        # Verificar que se removieron los duplicados
        assert len(result) < len(df)
    
    def test_get_equipment_types(self, sample_data):
        """Verifica que se retornan los tipos correctos."""
        treated = DataProcessor.treat_data(sample_data)
        types = DataProcessor.get_equipment_types(treated, 'EQUIPO_A')
        
        assert len(types) > 0
        assert 'MC' in types
    
    def test_get_equipment_types_empty_equipment(self, sample_data):
        """Verifica comportamiento con equipo no existente."""
        treated = DataProcessor.treat_data(sample_data)
        types = DataProcessor.get_equipment_types(treated, 'EQUIPO_NO_EXISTENTE')
        
        assert len(types) == 0
    
    def test_calculate_tbf_output_columns(self, sample_data):
        """Verifica que calculate_tbf retorna las columnas necesarias."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ['MC', 'MCE']
        result = DataProcessor.calculate_tbf(treated, 'EQUIPO_A', tipos)
        
        required_cols = ['Equipo', 'Fecha_Inicio', 'Fecha_Fin', 'Tipo', 'mdf', 'TBX', 'TTX']
        assert all(col in result.columns for col in required_cols)
    
    def test_calculate_tbf_creates_tbx_column(self, sample_data):
        """Verifica que TBX se calcula."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ['MC']
        result = DataProcessor.calculate_tbf(treated, 'EQUIPO_A', tipos)
        
        assert 'TBX' in result.columns
        # Primer TBX debe ser 0 (no hay evento anterior)
        assert result.loc[0, 'TBX'] == 0.0
    
    def test_calculate_tbf_preserves_data_integrity(self, sample_data):
        """Verifica que calculate_tbf no modifica datos innecesariamente."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ['MC', 'MCE']
        result = DataProcessor.calculate_tbf(treated, 'EQUIPO_A', tipos)
        
        # Verificar que no hay NaNs donde no deberían ser
        assert result['mdf'].notna().all()
        assert result['TTX'].notna().all()
    
    def test_calculate_tbf_multiple_equipment(self, sample_data):
        """Verifica que calculate_tbf filtra por equipo correctamente."""
        # Agregar segundo equipo
        new_data = sample_data.copy()
        new_data['Equipo'] = ['EQUIPO_B'] * 5
        combined = pd.concat([sample_data, new_data], ignore_index=True)
        
        treated = DataProcessor.treat_data(combined)
        tipos = ['MC']
        result = DataProcessor.calculate_tbf(treated, 'EQUIPO_A', tipos)
        
        # Verificar que solo devuelve EQUIPO_A
        assert result['Equipo'].unique() == ['EQUIPO_A']
    
    def test_treat_data_column_rename(self, sample_data):
        """Verifica que los renombres de columnas se realizan correctamente."""
        result = DataProcessor.treat_data(sample_data)
        
        # Verificar que 'Modo de Falla' fue renombrado a 'mdf'
        assert 'mdf' in result.columns
        assert 'Modo de Falla' not in result.columns
        
        # Verificar que 'Duracion' fue renombrado a 'TTX'
        assert 'TTX' in result.columns
        assert 'Duracion' not in result.columns
