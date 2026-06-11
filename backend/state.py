from typing import Optional
import pandas as pd
from src.reliability_analysis.core.filters import FilterManager
from src.reliability_analysis.core.data_processing import DataProcessor

current_data: Optional[pd.DataFrame] = None
filter_manager: Optional[FilterManager] = None
data_processor: Optional[DataProcessor] = None
