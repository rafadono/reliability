"""
Unit tests for DataProcessor.
"""

import pytest
import pandas as pd
from datetime import timedelta
from src.reliability_analysis.core.data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {
        "Fecha": [d.strftime("%d/%m/%Y") for d in dates],
        "Hora": ["08:00:00"] * 5,
        "Duracion": [1.0, 2.0, 1.5, 2.5, 1.0],
        "Tipo": ["MC", "MC", "MCE", "MC", "MP"],
        "Equipo": ["EQUIPO_A", "EQUIPO_A", "EQUIPO_A", "EQUIPO_A", "EQUIPO_A"],
        "Modo de Falla": ["Falla_1", "Falla_2", "Falla_1", "Falla_3", "Preventiva"],
    }
    return pd.DataFrame(data)


class TestDataProcessor:
    def test_treat_data_creates_required_columns(self, sample_data):
        """Verify that treat_data creates the necessary columns."""
        result = DataProcessor.treat_data(sample_data)

        required_cols = ["Start_Date", "End_Date", "TTX", "Type", "Equipment", "mdf"]
        assert all(col in result.columns for col in required_cols)

    def test_treat_data_date_parsing(self, sample_data):
        """Verify that dates are parsed correctly."""
        result = DataProcessor.treat_data(sample_data)

        assert pd.api.types.is_datetime64_any_dtype(result["Start_Date"])
        assert pd.api.types.is_datetime64_any_dtype(result["End_Date"])
        assert result["Start_Date"].iloc[0].year == 2024

    def test_treat_data_duration_calculation(self, sample_data):
        """Verify that End_Date is calculated correctly."""
        result = DataProcessor.treat_data(sample_data)

        for idx in result.index:
            expected_fin = result.loc[idx, "Start_Date"] + timedelta(
                hours=result.loc[idx, "TTX"]
            )
            assert result.loc[idx, "End_Date"] == expected_fin

    def test_treat_data_removes_duplicates(self):
        """Verify that duplicates are removed."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = {
            "Fecha": [d.strftime("%d/%m/%Y") for d in dates]
            + [dates[0].strftime("%d/%m/%Y")],
            "Hora": ["08:00:00"] * 4,
            "Duracion": [1.0, 2.0, 1.5, 1.0],
            "Tipo": ["MC", "MC", "MCE", "MC"],
            "Equipo": ["EQUIPO_A"] * 4,
            "Modo de Falla": ["Falla_1", "Falla_2", "Falla_1", "Falla_1"],
        }
        df = pd.DataFrame(data)
        result = DataProcessor.treat_data(df)

        # Verify duplicates are removed
        assert len(result) < len(df)

    def test_get_equipment_types(self, sample_data):
        """Verify that correct types are returned."""
        treated = DataProcessor.treat_data(sample_data)
        types = DataProcessor.get_equipment_types(treated, "EQUIPO_A")

        assert len(types) > 0
        assert "MC" in types

    def test_get_equipment_types_empty_equipment(self, sample_data):
        """Verify behavior with non-existing equipment."""
        treated = DataProcessor.treat_data(sample_data)
        types = DataProcessor.get_equipment_types(treated, "NON_EXISTENT_EQUIPMENT")

        assert len(types) == 0

    def test_calculate_tbf_output_columns(self, sample_data):
        """Verify that calculate_tbf returns necessary columns."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ["MC", "MCE"]
        result = DataProcessor.calculate_tbf(treated, "EQUIPO_A", tipos)

        required_cols = [
            "Equipment",
            "Start_Date",
            "End_Date",
            "Type",
            "mdf",
            "TBX",
            "TTX",
        ]
        assert all(col in result.columns for col in required_cols)

    def test_calculate_tbf_creates_tbx_column(self, sample_data):
        """Verify that TBX is calculated."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ["MC"]
        result = DataProcessor.calculate_tbf(treated, "EQUIPO_A", tipos)

        assert "TBX" in result.columns
        # First TBX should be 0 (no previous event)
        assert result.loc[0, "TBX"] == 0.0

    def test_calculate_tbf_preserves_data_integrity(self, sample_data):
        """Verify that calculate_tbf does not modify data unnecessarily."""
        treated = DataProcessor.treat_data(sample_data)
        tipos = ["MC", "MCE"]
        result = DataProcessor.calculate_tbf(treated, "EQUIPO_A", tipos)

        # Verify no NaNs where they shouldn't be
        assert result["mdf"].notna().all()
        assert result["TTX"].notna().all()

    def test_calculate_tbf_multiple_equipment(self, sample_data):
        """Verify that calculate_tbf filters by equipment correctly."""
        # Add second equipment
        new_data = sample_data.copy()
        new_data["Equipo"] = ["EQUIPO_B"] * 5
        combined = pd.concat([sample_data, new_data], ignore_index=True)

        treated = DataProcessor.treat_data(combined)
        tipos = ["MC"]
        result = DataProcessor.calculate_tbf(treated, "EQUIPO_A", tipos)

        # Verify it only returns EQUIPO_A
        assert (result["Equipment"].unique() == ["EQUIPO_A"]).all()

    def test_treat_data_column_rename(self, sample_data):
        """Verify that column renaming is executed correctly."""
        result = DataProcessor.treat_data(sample_data)

        # Verify that 'Modo de Falla' was renamed to 'mdf'
        assert "mdf" in result.columns
        assert "Modo de Falla" not in result.columns

        # Verify that 'Duracion' was renamed to 'TTX'
        assert "TTX" in result.columns
        assert "Duracion" not in result.columns
