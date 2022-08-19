# Imports
import os
import tqdm
import numpy as np
import pandas as pd



# Directories and filenames
data_dir = "data"
ph2_dir = os.path.join(data_dir, "ph2")
ph2_xlsx = "PH2_dataset.xlsx"


# Open PH2 XLSX file
ph2_df = pd.read_excel(os.path.join(ph2_dir, ph2_xlsx), skiprows=[i for i in range(12)])
print(ph2_df.head())
