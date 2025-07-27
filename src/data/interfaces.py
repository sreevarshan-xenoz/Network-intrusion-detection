"""
Base interfaces for data processing components.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np


class DatasetLoader(ABC):
    """Abstract base class for dataset loading."""
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file path."""
        pass
    
    @abstractmethod
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate data schema."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        pass


class PreprocessingPipeline(ABC):
    """Abstract base class for data preprocessing."""
    
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessing pipeline and transform data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        pass
    
    @abstractmethod
    def save_pipeline(self, path: str) -> None:
        """Save preprocessing pipeline to disk."""
        pass
    
    @abstractmethod
    def load_pipeline(self, path: str) -> None:
        """Load preprocessing pipeline from disk."""
        pass


class FeatureProcessor(ABC):
    """Abstract base class for feature processing components."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """Fit the processor to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)