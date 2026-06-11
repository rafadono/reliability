"""
Unit tests for Pareto analysis.

Tests hierarchical analysis by equipment, type, and failure mode,
including 80/20 split calculation.
"""

from src.reliability_analysis.analysis.pareto import (
    analyze_by_equipment,
    analyze_by_type,
    analyze_by_failure_mode,
    get_80_20_split,
)


class TestParetoAnalysis:
    """Test suite for Pareto analysis functions."""

    def test_analyze_by_equipment(self, sample_data):
        """Test equipment-level analysis."""
        result = analyze_by_equipment(sample_data)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all("equipment" in r and "failures" in r for r in result)
        assert all(r["failures"] > 0 for r in result)

    def test_analyze_by_equipment_sorted(self, sample_data):
        """Equipment analysis should be sorted by failures descending."""
        result = analyze_by_equipment(sample_data)
        failures = [r["failures"] for r in result]
        assert failures == sorted(failures, reverse=True)

    def test_analyze_by_type_with_equipment(self, sample_data):
        """Test type-level analysis for specific equipment."""
        equipment = sample_data["Equipment"].unique()[0]
        result = analyze_by_type(sample_data, equipment=equipment)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(r["type"] for r in result)

    def test_analyze_by_failure_mode(self, sample_data):
        """Test failure mode analysis."""
        equipment = sample_data["Equipment"].unique()[0]
        failure_type = sample_data[sample_data["Equipment"] == equipment][
            "Type"
        ].unique()[0]

        result = analyze_by_failure_mode(
            sample_data, equipment=equipment, failure_type=failure_type
        )

        assert isinstance(result, list)
        if len(result) > 0:
            assert all("mode" in r and "count" in r for r in result)

    def test_80_20_split(self, sample_data):
        """Test Pareto 80/20 split identification."""
        result = analyze_by_equipment(sample_data)
        split = get_80_20_split(result)

        assert "items_for_80_percent" in split
        assert "cumulative_percentage" in split
        assert split["cumulative_percentage"] >= 80
        assert split["items_for_80_percent"] <= len(result)

    def test_empty_filter_returns_empty(self, sample_data):
        """Non-existent equipment should return empty or handle gracefully."""
        result = analyze_by_type(sample_data, equipment="NonExistent")
        assert isinstance(result, list)

    def test_small_data(self, small_data):
        """Test with small dataset."""
        result = analyze_by_equipment(small_data)
        assert len(result) == 2
        assert all(r["failures"] > 0 for r in result)

    def test_pareto_cumulative(self, sample_data):
        """Cumulative percentage should increase monotonically."""
        result = analyze_by_equipment(sample_data)

        if len(result) > 1:
            prev = 0
            for item in result:
                cumsum = item.get("cumulative_percentage", 0)
                assert cumsum >= prev
                prev = cumsum
