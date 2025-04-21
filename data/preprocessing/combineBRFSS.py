import pandas as pd

# Dictionary mapping file names to years
csv_files = {
    "2019.csv": 2019,
    "2020.csv": 2020,
    "2021.csv": 2021,
    "2022.csv": 2022,
    "2023.csv": 2023
}

# Read and concatenate all CSVs while adding the "Year" column
df_list = [pd.read_csv(file, dtype=str).assign(Year=year) for file, year in csv_files.items()]
combined_df = pd.concat(df_list, ignore_index=True)

# Save merged dataset with Year column
combined_df.to_csv("brfss.csv", index=False)

print("Merged file saved successfully!")
