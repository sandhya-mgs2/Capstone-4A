import pandas as pd
import os

# Load SAS XPT file
xpt_file = "/Users/abhirijal/Documents/BRFSS/2019.XPT"  # Change filename as needed
df = pd.read_sas(xpt_file, format='xport')

# Save to CSV
df.to_csv("2019.csv", index=False)
