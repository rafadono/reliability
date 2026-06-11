"""
Unit tests for Jackknife confidence interval calculation.

Tests Leave-One-Out resampling and multi-parameter CI generation.
"""

import pytest
import numpy as np
from src.reliability_analysis.analysis.jackknife import (
    calculate_jackknife_ci,
    calculate_multi_parameter_ci,
    bootstrap_ci,
)


class TestJackknifeAnalysis:
    """Test suite for Jackknife confidence interval calculations."""

    def test_jackknife_ci_basic(self):
        """Test basic Jackknife CI calculation."""
        data = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        
        result = calculate_jackknife_ci(data, confidence=0.95)
        
        assert 'mean' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['mean'] < result['ci_upper']

    def test_jackknife_ci_small_sample(self):
        """Test with minimum viable sample size."""
        data = np.array([100, 120, 140])
        
        result = calculate_jackknife_ci(data, confidence=0.95)
        
        assert result['ci_lower'] >= 0
        assert result['ci_upper'] > result['ci_lower']

    def test_jackknife_ci_confidence_levels(self):
        """Higher confidence should give wider intervals."""
        data = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        
        ci_90 = calculate_jackknife_ci(data, confidence=0.90)
        ci_95 = calculate_jackknife_ci(data, confidence=0.95)
        ci_99 = calculate_jackknife_ci(data, confidence=0.99)
        
        width_90 = ci_90['ci_upper'] - ci_90['ci_lower']
        width_95 = ci_95['ci_upper'] - ci_95['ci_lower']
        width_99 = ci_99['ci_upper'] - ci_99['ci_lower']
        
        assert width_90 <= width_95 <= width_99

    def test_multi_parameter_ci(self):
        """Test multi-parameter CI calculation."""
        data = {
            'weibull_alpha': np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190]),
            'weibull_beta': np.array([1.5, 1.6, 1.5, 1.7, 1.6, 1.5, 1.8, 1.7, 1.6, 1.5]),
        }
        
        result = calculate_multi_parameter_ci(data, confidence=0.95)
        
        assert 'weibull_alpha' in result
        assert 'weibull_beta' in result
        assert all('mean' in result[key] and 'ci_lower' in result[key] 
                  for key in result)

    def test_bootstrap_ci(self):
        """Test bootstrap CI as alternative."""
        data = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        
        result = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)
        
        assert 'mean' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['mean'] < result['ci_upper']

    def test_jackknife_vs_bootstrap(self):
        """Jackknife and Bootstrap should give similar results."""
        data = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        
        jk = calculate_jackknife_ci(data, confidence=0.95)
        bs = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)
        
        # Means should be very close
        assert abs(jk['mean'] - bs['mean']) < 5
        # Intervals should overlap
        assert jk['ci_lower'] < bs['ci_upper'] and bs['ci_lower'] < jk['ci_upper']

    def test_ci_contains_true_mean(self, small_data):
        """CI should contain the sample mean in most cases."""
        dias = small_data['Dias'].values.astype(float)
        
        result = calculate_jackknife_ci(dias, confidence=0.95)
        sample_mean = np.mean(dias)
        
        assert result['ci_lower'] <= sample_mean <= result['ci_upper']

    def test_constant_data_ci(self):
        """Constant data should have zero-width CI."""
        data = np.array([100, 100, 100, 100, 100])
        
        result = calculate_jackknife_ci(data, confidence=0.95)
        
        assert abs(result['mean'] - 100) < 0.01
        assert result['ci_lower'] == result['ci_upper'] or \
               abs(result['ci_upper'] - result['ci_lower']) < 1

    def test_outlier_handling(self):
        """Test that outliers don't break CI calculation."""
        data = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 1000])
        
        result = calculate_jackknife_ci(data, confidence=0.95)
        
        assert result['ci_lower'] > 0
        assert result['ci_upper'] > result['ci_lower']
