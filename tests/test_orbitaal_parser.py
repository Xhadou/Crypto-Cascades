"""Tests for ORBITAAL data parsing and validation."""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime

from src.preprocessing.orbitaal_parser import OrbitaalParser
from src.utils.exceptions import DataLoadError, DataValidationError, InsufficientDataError


class TestOrbitaalParserInit:
    """Tests for OrbitaalParser initialization."""

    def test_creates_instance(self):
        parser = OrbitaalParser()
        assert parser is not None

    def test_default_data_dir(self):
        parser = OrbitaalParser()
        assert parser.data_dir == Path("data/raw/orbitaal")

    def test_custom_data_dir(self):
        parser = OrbitaalParser(data_dir="/tmp/custom")
        assert parser.data_dir == Path("/tmp/custom")

    def test_has_column_mapping(self):
        parser = OrbitaalParser()
        assert "SRC_ID" in parser.COLUMN_MAPPING
        assert "DST_ID" in parser.COLUMN_MAPPING
        assert "TIMESTAMP" in parser.COLUMN_MAPPING
        assert "VALUE_SATOSHI" in parser.COLUMN_MAPPING

    def test_snapshot_columns_defined(self):
        assert "source_id" in OrbitaalParser.SNAPSHOT_COLUMNS
        assert "target_id" in OrbitaalParser.SNAPSHOT_COLUMNS
        assert "btc_value" in OrbitaalParser.SNAPSHOT_COLUMNS

    def test_stream_columns_defined(self):
        assert "timestamp" in OrbitaalParser.STREAM_COLUMNS
        assert "source_id" in OrbitaalParser.STREAM_COLUMNS
        assert "target_id" in OrbitaalParser.STREAM_COLUMNS


class TestStandardizeColumns:
    """Tests for column name standardization."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    def test_renames_orbitaal_columns(self, parser):
        df = pd.DataFrame({
            "SRC_ID": [1, 2],
            "DST_ID": [3, 4],
            "VALUE_SATOSHI": [100_000_000, 200_000_000],
            "VALUE_USD": [5000.0, 10000.0],
        })
        result = parser._standardize_columns(df)
        assert "source_id" in result.columns
        assert "target_id" in result.columns
        assert "btc_value" in result.columns
        assert "usd_value" in result.columns

    def test_converts_satoshi_to_btc(self, parser):
        """Values above 1e6 are assumed to be satoshi and converted to BTC."""
        df = pd.DataFrame({
            "SRC_ID": [1],
            "DST_ID": [2],
            "VALUE_SATOSHI": [100_000_000],  # 1 BTC in satoshi
        })
        result = parser._standardize_columns(df)
        assert result["btc_value"].iloc[0] == pytest.approx(1.0)

    def test_does_not_convert_small_btc_values(self, parser):
        """Values already in BTC (below 1e6) should not be converted."""
        df = pd.DataFrame({
            "source_id": [1],
            "target_id": [2],
            "btc_value": [1.5],
        })
        result = parser._standardize_columns(df)
        assert result["btc_value"].iloc[0] == pytest.approx(1.5)

    def test_passes_through_already_standard_columns(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [3, 4],
            "btc_value": [0.5, 1.0],
        })
        result = parser._standardize_columns(df)
        assert list(result.columns) == ["source_id", "target_id", "btc_value"]

    def test_renames_source_target_shorthand(self, parser):
        df = pd.DataFrame({
            "source": [1],
            "target": [2],
        })
        result = parser._standardize_columns(df)
        assert "source_id" in result.columns
        assert "target_id" in result.columns


class TestValidateTransactions:
    """Tests for transaction validation."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [4, 5, 6],
            "btc_value": [1.0, 2.0, 3.0],
            "usd_value": [5000.0, 10000.0, 15000.0],
            "timestamp": [1508000000, 1508001000, 1508002000],
        })

    def test_valid_data_passes(self, parser, valid_df):
        is_valid, issues, df_clean = parser.validate_transactions(valid_df)
        assert is_valid is True
        assert len(issues) == 0
        assert len(df_clean) == len(valid_df)

    def test_missing_required_columns_fails(self, parser):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        is_valid, issues, _ = parser.validate_transactions(df)
        assert is_valid is False
        assert any("CRITICAL" in i for i in issues)

    def test_missing_source_id_fails(self, parser):
        df = pd.DataFrame({"target_id": [1, 2], "btc_value": [1.0, 2.0]})
        is_valid, issues, _ = parser.validate_transactions(df)
        assert is_valid is False

    def test_missing_target_id_fails(self, parser):
        df = pd.DataFrame({"source_id": [1, 2], "btc_value": [1.0, 2.0]})
        is_valid, issues, _ = parser.validate_transactions(df)
        assert is_valid is False

    def test_removes_null_source_ids_non_strict(self, parser):
        df = pd.DataFrame({
            "source_id": [1, None, 3],
            "target_id": [4, 5, 6],
            "btc_value": [1.0, 2.0, 3.0],
        })
        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)
        assert is_valid is True
        assert len(df_clean) == 2

    def test_removes_self_loops_non_strict(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [1, 5, 6],  # first row is a self-loop
            "btc_value": [1.0, 2.0, 3.0],
        })
        is_valid, issues, df_clean = parser.validate_transactions(df, strict=False)
        assert is_valid is True
        assert len(df_clean) == 2
        assert any("self-loop" in i for i in issues)

    def test_removes_negative_btc_values_non_strict(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [4, 5, 6],
            "btc_value": [1.0, -2.0, 3.0],
        })
        _, issues, df_clean = parser.validate_transactions(df, strict=False)
        assert len(df_clean) == 2
        assert any("negative BTC" in i for i in issues)

    def test_removes_negative_usd_values_non_strict(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [3, 4],
            "usd_value": [100.0, -50.0],
        })
        _, issues, df_clean = parser.validate_transactions(df, strict=False)
        assert len(df_clean) == 1
        assert any("negative USD" in i for i in issues)

    def test_warns_extremely_large_btc(self, parser):
        df = pd.DataFrame({
            "source_id": [1],
            "target_id": [2],
            "btc_value": [2_000_000.0],  # More than 1M BTC
        })
        _, issues, _ = parser.validate_transactions(df)
        assert any("Extremely large BTC" in i for i in issues)

    def test_warns_invalid_timestamps(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [3, 4],
            "timestamp": [100, 1508000000],  # First is before Bitcoin genesis
        })
        _, issues, _ = parser.validate_transactions(df)
        assert any("invalid timestamps" in i for i in issues)

    def test_removes_duplicate_rows_non_strict(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 1, 2],
            "target_id": [3, 3, 4],
            "btc_value": [1.0, 1.0, 2.0],
        })
        _, issues, df_clean = parser.validate_transactions(df, strict=False)
        assert len(df_clean) == 2
        assert any("duplicate" in i for i in issues)

    def test_strict_mode_reports_but_does_not_remove(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [1, 5, 6],  # self-loop on row 0
            "btc_value": [1.0, 2.0, 3.0],
        })
        # strict=True should not remove rows, but is_valid depends on issues
        is_valid, issues, df_clean = parser.validate_transactions(df, strict=True)
        # In strict mode, issues exist so is_valid should be False
        assert is_valid is False
        # The cleaned df should still have all rows in strict mode
        assert len(df_clean) == 3

    def test_warns_no_value_or_timestamp_columns(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [3, 4],
        })
        _, issues, _ = parser.validate_transactions(df)
        assert any("No value or timestamp" in i for i in issues)

    def test_summary_shows_removal_count(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [1, 5, 6],  # self-loop
            "btc_value": [1.0, 2.0, 3.0],
        })
        _, issues, _ = parser.validate_transactions(df, strict=False)
        assert any("SUMMARY" in i for i in issues)


class TestLoadSnapshot:
    """Tests for loading snapshot files."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    def test_raises_on_missing_file(self, parser):
        with pytest.raises(DataLoadError):
            parser.load_snapshot("/nonexistent/path/file.csv")

    def test_loads_csv_file(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_snapshot(str(csv_path))
        assert len(result) == 20
        assert "source_id" in result.columns

    def test_loads_parquet_file(self, parser, tmp_path):
        pq_path = tmp_path / "snapshot.parquet"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_parquet(pq_path, index=False)
        result = parser.load_snapshot(str(pq_path))
        assert len(result) == 20

    def test_standardizes_orbitaal_column_names(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        df = pd.DataFrame({
            "SRC_ID": np.arange(1, 21),
            "DST_ID": np.arange(21, 41),
            "VALUE_SATOSHI": np.full(20, 100_000_000),
            "VALUE_USD": np.full(20, 5000.0),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_snapshot(str(csv_path))
        assert "source_id" in result.columns
        assert "target_id" in result.columns
        assert "btc_value" in result.columns

    def test_filters_by_min_btc_value(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "btc_value": [0.001] * 10 + [5.0] * 10,
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_snapshot(str(csv_path), min_btc_value=1.0)
        assert len(result) == 10

    def test_filters_by_min_usd_value(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": [10.0] * 10 + [1000.0] * 10,
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_snapshot(str(csv_path), min_usd_value=500.0)
        assert len(result) == 10

    def test_raises_on_insufficient_data(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        df = pd.DataFrame({
            "source_id": [1, 2, 3],
            "target_id": [4, 5, 6],
            "btc_value": [1.0, 2.0, 3.0],
            "usd_value": [100.0, 200.0, 300.0],
        })
        df.to_csv(csv_path, index=False)
        with pytest.raises(InsufficientDataError):
            parser.load_snapshot(str(csv_path))

    def test_skips_validation_when_disabled(self, parser, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        # Include self-loops; with validate=False they should remain
        df = pd.DataFrame({
            "source_id": list(range(1, 21)),
            "target_id": list(range(1, 21)),  # all self-loops
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_snapshot(str(csv_path), validate=False)
        assert len(result) == 20


class TestLoadStream:
    """Tests for loading stream graph files."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    def test_raises_on_missing_file(self, parser):
        with pytest.raises(DataLoadError):
            parser.load_stream("/nonexistent/path/stream.csv")

    def test_loads_stream_csv(self, parser, tmp_path):
        csv_path = tmp_path / "stream.csv"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "timestamp": np.linspace(1508000000, 1508100000, 20, dtype=int),
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_stream(str(csv_path))
        assert "datetime" in result.columns
        assert "date" in result.columns
        assert "hour" in result.columns

    def test_converts_timestamp_to_datetime(self, parser, tmp_path):
        csv_path = tmp_path / "stream.csv"
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "timestamp": np.full(20, 1508000000),
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_stream(str(csv_path))
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_filters_by_time_range(self, parser, tmp_path):
        csv_path = tmp_path / "stream.csv"
        base_ts = 1508000000
        timestamps = [base_ts + i * 86400 for i in range(20)]  # 20 days
        df = pd.DataFrame({
            "source_id": np.arange(1, 21),
            "target_id": np.arange(21, 41),
            "timestamp": timestamps,
            "btc_value": np.random.uniform(0.01, 10.0, 20),
            "usd_value": np.random.uniform(100, 50000, 20),
        })
        df.to_csv(csv_path, index=False)
        start = datetime(2017, 10, 15)
        end = datetime(2017, 10, 20)
        result = parser.load_stream(str(csv_path), start_time=start, end_time=end)
        assert all(result["datetime"] >= start)
        assert all(result["datetime"] <= end)

    def test_standardizes_orbitaal_columns_for_stream(self, parser, tmp_path):
        csv_path = tmp_path / "stream.csv"
        df = pd.DataFrame({
            "SRC_ID": np.arange(1, 21),
            "DST_ID": np.arange(21, 41),
            "TIMESTAMP": np.linspace(1508000000, 1508100000, 20, dtype=int),
            "VALUE_SATOSHI": np.full(20, 100_000_000),
            "VALUE_USD": np.full(20, 5000.0),
        })
        df.to_csv(csv_path, index=False)
        result = parser.load_stream(str(csv_path))
        assert "source_id" in result.columns
        assert "target_id" in result.columns
        assert result["btc_value"].iloc[0] == pytest.approx(1.0)


class TestComputeWalletActivity:
    """Tests for wallet activity computation."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def tx_df(self):
        return pd.DataFrame({
            "source_id": [1, 1, 2, 3],
            "target_id": [2, 3, 3, 1],
            "btc_value": [1.0, 0.5, 2.0, 0.1],
            "usd_value": [5000.0, 2500.0, 10000.0, 500.0],
        })

    @pytest.fixture
    def tx_df_with_time(self):
        return pd.DataFrame({
            "source_id": [1, 1, 2, 3],
            "target_id": [2, 3, 3, 1],
            "btc_value": [1.0, 0.5, 2.0, 0.1],
            "usd_value": [5000.0, 2500.0, 10000.0, 500.0],
            "datetime": pd.to_datetime([
                "2017-10-01", "2017-10-02", "2017-10-03", "2017-10-04"
            ]),
        })

    def test_returns_dataframe(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        assert isinstance(activity, pd.DataFrame)

    def test_all_wallets_present(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        wallet_ids = set(activity["wallet_id"].values)
        assert wallet_ids == {1, 2, 3}

    def test_net_btc_computed(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        assert "net_btc" in activity.columns
        assert "net_usd" in activity.columns

    def test_total_tx_computed(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        assert "total_tx" in activity.columns
        # Wallet 1: 2 outgoing + 1 incoming = 3 total
        w1 = activity[activity["wallet_id"] == 1].iloc[0]
        assert w1["total_tx"] == 3

    def test_btc_in_out_correct(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        w1 = activity[activity["wallet_id"] == 1].iloc[0]
        # Wallet 1 sends 1.0 + 0.5 = 1.5 BTC out, receives 0.1 BTC in
        assert w1["btc_out"] == pytest.approx(1.5)
        assert w1["btc_in"] == pytest.approx(0.1)
        assert w1["net_btc"] == pytest.approx(0.1 - 1.5)

    def test_includes_time_columns_when_available(self, parser, tx_df_with_time):
        activity = parser.compute_wallet_activity(tx_df_with_time)
        assert "first_activity" in activity.columns
        assert "last_activity" in activity.columns

    def test_no_time_columns_without_datetime(self, parser, tx_df):
        activity = parser.compute_wallet_activity(tx_df)
        assert "first_activity" not in activity.columns

    def test_wallets_only_receiving_have_zero_outgoing(self, parser):
        df = pd.DataFrame({
            "source_id": [1],
            "target_id": [2],
            "btc_value": [5.0],
            "usd_value": [25000.0],
        })
        activity = parser.compute_wallet_activity(df)
        w2 = activity[activity["wallet_id"] == 2].iloc[0]
        assert w2["btc_out"] == 0.0
        assert w2["tx_out_count"] == 0


class TestIdentifyActiveWallets:
    """Tests for active wallet identification."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def tx_df(self):
        return pd.DataFrame({
            "source_id": [1, 1, 1, 2, 3],
            "target_id": [2, 3, 4, 4, 4],
            "btc_value": [1.0, 0.5, 2.0, 0.1, 0.01],
            "usd_value": [5000.0, 2500.0, 10000.0, 500.0, 50.0],
        })

    def test_returns_set(self, parser, tx_df):
        active = parser.identify_active_wallets(tx_df, min_transactions=1)
        assert isinstance(active, set)

    def test_filters_by_min_transactions(self, parser, tx_df):
        active = parser.identify_active_wallets(tx_df, min_transactions=3)
        # Only wallet 1 (3 out) and wallet 4 (3 in) have >= 3 transactions
        assert 1 in active
        assert 4 in active

    def test_filters_by_min_btc_volume(self, parser, tx_df):
        active = parser.identify_active_wallets(
            tx_df, min_transactions=1, min_btc_volume=2.0
        )
        # Only wallets with total volume >= 2.0 BTC
        assert len(active) > 0

    def test_all_wallets_with_low_threshold(self, parser, tx_df):
        active = parser.identify_active_wallets(tx_df, min_transactions=1)
        all_wallets = set(tx_df["source_id"]) | set(tx_df["target_id"])
        assert active == all_wallets


class TestCreateTemporalSnapshots:
    """Tests for temporal snapshot creation."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def stream_df(self):
        return pd.DataFrame({
            "source_id": list(range(1, 11)),
            "target_id": list(range(11, 21)),
            "btc_value": np.random.uniform(0.01, 10.0, 10),
            "usd_value": np.random.uniform(100, 50000, 10),
            "datetime": pd.date_range("2017-10-01", periods=10, freq="D"),
        })

    def test_creates_daily_snapshots(self, parser, stream_df):
        snapshots = parser.create_temporal_snapshots(stream_df, frequency="D")
        assert isinstance(snapshots, dict)
        assert len(snapshots) == 10  # 10 days

    def test_creates_weekly_snapshots(self, parser, stream_df):
        snapshots = parser.create_temporal_snapshots(stream_df, frequency="W")
        assert isinstance(snapshots, dict)
        assert len(snapshots) <= 3  # ~10 days fits in 2-3 weeks

    def test_raises_on_missing_time_column(self, parser):
        df = pd.DataFrame({
            "source_id": [1, 2],
            "target_id": [3, 4],
            "btc_value": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="not found"):
            parser.create_temporal_snapshots(df)

    def test_snapshot_values_are_dataframes(self, parser, stream_df):
        snapshots = parser.create_temporal_snapshots(stream_df, frequency="D")
        for key, value in snapshots.items():
            assert isinstance(value, pd.DataFrame)
            assert "source_id" in value.columns


class TestComputeTransactionVolumeByTime:
    """Tests for transaction volume aggregation."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def stream_df(self):
        return pd.DataFrame({
            "source_id": [1, 1, 2, 2, 3],
            "target_id": [2, 3, 3, 4, 4],
            "btc_value": [1.0, 0.5, 2.0, 0.1, 3.0],
            "usd_value": [5000.0, 2500.0, 10000.0, 500.0, 15000.0],
            "datetime": pd.to_datetime([
                "2017-10-01 10:00",
                "2017-10-01 14:00",
                "2017-10-02 09:00",
                "2017-10-02 15:00",
                "2017-10-03 11:00",
            ]),
        })

    def test_returns_dataframe_with_expected_columns(self, parser, stream_df):
        vol = parser.compute_transaction_volume_by_time(stream_df, frequency="D")
        assert "tx_count" in vol.columns
        assert "btc_volume" in vol.columns
        assert "usd_volume" in vol.columns
        assert "unique_senders" in vol.columns
        assert "unique_receivers" in vol.columns

    def test_daily_aggregation(self, parser, stream_df):
        vol = parser.compute_transaction_volume_by_time(stream_df, frequency="D")
        assert len(vol) == 3  # 3 distinct days

    def test_raises_on_missing_time_column(self, parser):
        df = pd.DataFrame({
            "source_id": [1],
            "target_id": [2],
            "btc_value": [1.0],
            "usd_value": [5000.0],
        })
        with pytest.raises(ValueError, match="not found"):
            parser.compute_transaction_volume_by_time(df)


class TestGetTopWallets:
    """Tests for top wallet retrieval."""

    @pytest.fixture
    def parser(self):
        return OrbitaalParser()

    @pytest.fixture
    def tx_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "source_id": np.random.randint(0, 50, 200),
            "target_id": np.random.randint(50, 100, 200),
            "btc_value": np.random.lognormal(0, 2, 200),
            "usd_value": np.random.lognormal(8, 2, 200),
        })

    def test_returns_dataframe(self, parser, tx_df):
        top = parser.get_top_wallets(tx_df, metric="net_btc", n=10)
        assert isinstance(top, pd.DataFrame)
        assert len(top) == 10

    def test_raises_on_unknown_metric(self, parser, tx_df):
        with pytest.raises(ValueError, match="Unknown metric"):
            parser.get_top_wallets(tx_df, metric="nonexistent_column")

    def test_ascending_order(self, parser, tx_df):
        top = parser.get_top_wallets(tx_df, metric="net_btc", n=5, ascending=True)
        assert len(top) == 5
        # Ascending means smallest first
        assert top["net_btc"].iloc[0] <= top["net_btc"].iloc[-1]

    def test_default_descending_order(self, parser, tx_df):
        top = parser.get_top_wallets(tx_df, metric="total_tx", n=5)
        assert top["total_tx"].iloc[0] >= top["total_tx"].iloc[-1]
