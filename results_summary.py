from devcode.blocks import ResultHandler

from load_dataset import get_datasets

import warnings
warnings.filterwarnings('ignore')

datasets = get_datasets()

ResultHandler.global_ols_summary(datasets)        # Global Ordinary Least Square (G-OLS)
ResultHandler.global_lssvm_summary(datasets)      # Global Least Square Support Vector Machine (G-LSSVM)

ResultHandler.local_lssvm_summary(datasets)       # Local Least Square Support Vector Machine (L-LSSVM)

ResultHandler.regional_ols_summary(datasets)      # Regional Ordinary Least Square (R-OLS)
ResultHandler.regional_lssvm_summary(datasets)    # Regional Least Square Support Vector Machine (R-LSSVM)

