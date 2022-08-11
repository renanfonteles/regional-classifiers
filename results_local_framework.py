from devcode.analysis.results import process_results
from devcode.utils import load_csv_as_pandas

global_results_file = f"results/local-results/cbic/temp_glssvm_cbic"
df_results = load_csv_as_pandas(global_results_file)

process_results(df_results)
