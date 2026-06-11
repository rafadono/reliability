"""
Pruebas unitarias para ReliabilityFitter y KijimaMontecarlo.
"""

import pytest
import numpy as np
import pandas as pd
from src.reliability_analysis.analysis.models import ReliabilityFitter, KijimaMontecarlo


@pytest.fixture
def sample_fit_data():
    """Crea datos de ejemplo para fitting."""
    np.random.seed(42)
    tbx_data = np.random.weibull(2, 30) * 100  # Datos Weibull
    ttx_data = np.random.weibull(1.5, 30) * 50
    
    data = {
        'TBX': tbx_data,
        'TTX': ttx_data,
        'mdf': ['Falla_A'] * 20 + ['Falla_B'] * 10
    }
    return pd.DataFrame(data)


class TestReliabilityFitter:
    
    def test_fitter_initialization(self):
        """Verifica que ReliabilityFitter se inicializa correctamente."""
        fitter = ReliabilityFitter()
        assert fitter.excluded_models is not None
        assert len(fitter.excluded_models) > 0
    
    def test_fit_returns_dict(self, sample_fit_data):
        """Verifica que fit retorna un diccionario."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        assert isinstance(result, dict)
    
    def test_fit_required_keys(self, sample_fit_data):
        """Verifica que fit retorna todas las claves necesarias."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        required_keys = ['best_distribution', 'name', 'parameters', 'mean', 'std_dev', 'AICc', 'BIC']
        assert all(key in result for key in required_keys)
    
    def test_fit_distribution_name_not_empty(self, sample_fit_data):
        """Verifica que se retorna un nombre de distribución."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        assert isinstance(result['name'], str)
        assert len(result['name']) > 0
    
    def test_fit_mean_is_positive(self, sample_fit_data):
        """Verifica que la media es positiva."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        assert result['mean'] > 0
    
    def test_fit_std_dev_is_positive(self, sample_fit_data):
        """Verifica que la desviación estándar es positiva."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        assert result['std_dev'] > 0
    
    def test_fit_metrics_are_numbers(self, sample_fit_data):
        """Verifica que AIC y BIC son números."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_B'])
        
        assert isinstance(result['AICc'], (int, float)) and not np.isnan(result['AICc'])
        assert isinstance(result['BIC'], (int, float)) and not np.isnan(result['BIC'])
    
    def test_fit_with_all_censored(self, sample_fit_data):
        """Verifica comportamiento cuando todos los datos son censurados."""
        fitter = ReliabilityFitter()
        
        # Si intentamos censurar todos, debería usar solo los no censurados
        result = fitter.fit(sample_fit_data, 'TBX', ['Falla_A'])
        
        assert result['name'] is not None
    
    def test_fit_ttx_column(self, sample_fit_data):
        """Verifica que fit funciona con columna TTX."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, 'TTX', ['Falla_B'])
        
        assert result['mean'] > 0
        assert result['std_dev'] > 0


class TestKijimaMontecarlo:
    
    def test_kijima_initialization(self):
        """Verifica que KijimaMontecarlo se inicializa."""
        kijima = KijimaMontecarlo()
        assert kijima.models is not None
    
    def test_fit_single_model_returns_dict(self, sample_fit_data):
        """Verifica que fit con modelo único retorna dict."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert isinstance(result, dict)
    
    def test_fit_multiple_models_returns_list(self, sample_fit_data):
        """Verifica que fit con múltiples modelos retorna lista."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=[1, 2])
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_fit_required_keys_single_model(self, sample_fit_data):
        """Verifica claves en resultado de modelo único."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        required_keys = ['model_name', 'beta', 'eta', 'ar', 'ap', 'AIC', 'BIC', 'mean', 'std', 't', 'R', 'failure_rate']
        assert all(key in result for key in required_keys)
    
    def test_fit_beta_eta_positive(self, sample_fit_data):
        """Verifica que beta y eta son positivos."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert result['beta'] > 0
        assert result['eta'] > 0
    
    def test_fit_ar_ap_in_range(self, sample_fit_data):
        """Verifica que ar y ap están en rango [0, 1]."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert 0 <= result['ar'] <= 1
        assert 0 <= result['ap'] <= 1
    
    def test_fit_mean_positive(self, sample_fit_data):
        """Verifica que la media es positiva."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert result['mean'] > 0
    
    def test_fit_curves_same_length(self, sample_fit_data):
        """Verifica que R, failure_rate y t tienen la misma longitud."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert len(result['t']) == len(result['R']) == len(result['failure_rate'])
    
    def test_fit_reliability_in_range(self, sample_fit_data):
        """Verifica que confiabilidad está en [0, 1]."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert np.all((result['R'] >= 0) & (result['R'] <= 1))
    
    def test_fit_failure_rate_positive(self, sample_fit_data):
        """Verifica que la tasa de falla es positiva."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=1)
        
        assert np.all(result['failure_rate'] >= 0)
    
    def test_model_names_correct(self, sample_fit_data):
        """Verifica que los nombres de modelos son correctos."""
        kijima = KijimaMontecarlo()
        result = kijima.fit(sample_fit_data, 'TBX', ['Falla_B'], modelos=[1, 2])
        
        assert result[0]['model_name'] == 'Kijima I'
        assert result[1]['model_name'] == 'Kijima II'
