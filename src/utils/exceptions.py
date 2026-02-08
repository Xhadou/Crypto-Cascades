"""
Custom Exceptions for Crypto Cascades

Provides specific exception types for better error handling and debugging.
"""


class CryptoCascadesError(Exception):
    """Base exception for all Crypto Cascades errors."""
    pass


class DataLoadError(CryptoCascadesError):
    """Error loading or parsing data files."""
    def __init__(self, filepath: str, reason: str):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to load {filepath}: {reason}")


class DataValidationError(CryptoCascadesError):
    """Error validating data integrity."""
    def __init__(self, issues: list):
        self.issues = issues
        super().__init__(f"Data validation failed: {', '.join(issues[:3])}")


class InsufficientDataError(CryptoCascadesError):
    """Not enough data for analysis."""
    def __init__(self, required: int, available: int, data_type: str = "records"):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data: need {required} {data_type}, have {available}"
        )


class ModelFittingError(CryptoCascadesError):
    """Error fitting model parameters."""
    def __init__(self, model: str, reason: str):
        self.model = model
        self.reason = reason
        super().__init__(f"Failed to fit {model}: {reason}")


class HypothesisTestError(CryptoCascadesError):
    """Error running hypothesis test."""
    def __init__(self, hypothesis: str, reason: str):
        self.hypothesis = hypothesis
        self.reason = reason
        super().__init__(f"Hypothesis {hypothesis} test failed: {reason}")


class ConfigurationError(CryptoCascadesError):
    """Error in configuration."""
    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(f"Configuration error for '{key}': {reason}")
