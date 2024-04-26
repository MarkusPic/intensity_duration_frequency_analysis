from idf_analysis.idf_class import IntensityDurationFrequencyAnalyse
import pandas as pd

# Load the IDF table
idf_table_path = "ehyd_112086_idf_data/idf_table_UNIX.csv"
idf_table = pd.read_csv(idf_table_path, header=0, index_col=0)


