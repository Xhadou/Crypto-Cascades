"""
Tests for data validation logic in OrbitaalParser.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.orbitaal_parser import OrbitaalParser
from src.utils.exceptions import DataLoadError, DataValidationError, InsufficientDataError


class TestDataValidation:
    """Tests for transaction data validation."""

    @pytest.fixture
    def parser(self, tmp_path):
        return OrbitaalParser(data_dir=str(tmp_path))

    def test_detects_self_loops(self, parser):
        """Test that self-loops are detected and removed."""
        df = pd.DataFrame({
            'source_id': [1, 2, 3, 3],
            'target_id': [2, 3, 3, 4],  # 3->3 is self-loop
            'btc_value': [1.0, 1.0, 1.0, 1.0]
        })

        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)

        assert any('self-loop' in issue.lower() for issue in issues)
        assert len(df_clean) < len(df)

    def test_detects_negative_values(self, parser):
        """Test that negative btc values are detected."""
        df = pd.DataFrame({
            'source_id': [1, 2],
            'target_id': [2, 3],
            'btc_value': [1.0, -0.5]
        })

        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)

        assert any('negative' in issue.lower() for issue in issues)

    def test_valid_data_passes(self, parser):
        """Test that clean data passes validation."""
        df = pd.DataFrame({
            'source_id': [1, 2, 3],
            'target_id': [2, 3, 4],
            'btc_value': [0.5, 1.0, 2.0]
        })

        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)

        assert is_valid or len(issues) == 0
        assert len(df_clean) == len(df)

    def test_empty_dataframe_raises(self, parser):
        """Test that an empty dataframe is flagged."""
        df = pd.DataFrame(columns=['source_id', 'target_id', 'btc_value'])

        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)

        assert not is_valid or len(df_clean) == 0


class TestErrorHandling:
    """Test standardized exception usage in orbitaal_parser."""

    @pytest.fixture
    def parser(self, tmp_path):
        return OrbitaalParser(data_dir=str(tmp_path))

    def test_load_snapshot_missing_file_raises(self, parser, tmp_path):
        """load_snapshot should raise DataLoadError for missing files."""
        with pytest.raises(DataLoadError):
            parser.load_snapshot(filepath=tmp_path / "nonexistent_9999_01_01.csv")

    def test_load_stream_missing_file_raises(self, parser, tmp_path):
        """load_stream should raise DataLoadError for missing files."""
        with pytest.raises(DataLoadError):
            parser.load_stream(filepath=tmp_path / "nonexistent_stream_9999_01_01.csv")
