import pandas as pd
import numpy as np

##############################################
# STEP 1: Process CDC Data for Depression & Anxiety
##############################################

# Load CDC data (assumes file "cdc_reduced.csv")
cdc_df = pd.read_csv("cdc_reduced.csv")

# Convert the "Time Period Start Date" to datetime and extract Year and Month.
cdc_df['StartDate'] = pd.to_datetime(cdc_df['Time Period Start Date'], errors='coerce')
cdc_df['Year'] = cdc_df['StartDate'].dt.year
cdc_df['Month'] = cdc_df['StartDate'].dt.month

# --- CDC for Depression ---
# For depression in CDC, use rows where the indicator applies to depression.
cdc_dep_indicators = ["Symptoms of Depressive Disorder", "Symptoms of Anxiety Disorder or Depressive Disorder"]
cdc_dep = cdc_df[cdc_df['Indicator'].isin(cdc_dep_indicators)].copy()

# Group by State, Year, Month and compute the average Value.
cdc_dep_grouped = (
    cdc_dep.groupby(['State', 'Year', 'Month'], as_index=False)
           .agg({'Value': 'mean'})
)
cdc_dep_grouped.rename(columns={'Value': 'CDC_Depression_Value'}, inplace=True)

# --- CDC for Anxiety ---
# For anxiety in CDC, use rows where the indicator applies to anxiety.
cdc_anx_indicators = ["Symptoms of Anxiety Disorder"]
cdc_anx = cdc_df[cdc_df['Indicator'].isin(cdc_anx_indicators)].copy()

cdc_anx_grouped = (
    cdc_anx.groupby(['State', 'Year', 'Month'], as_index=False)
           .agg({'Value': 'mean'})
)
cdc_anx_grouped.rename(columns={'Value': 'CDC_Anxiety_Value'}, inplace=True)

##############################################
# STEP 2: Process BRFSS Data for Depression
##############################################

# Load BRFSS data (assumes file "brfss_reduced.csv")
brfss_df = pd.read_csv("brfss_reduced.csv")

# Clean the IYEAR column by removing extra characters (e.g., "b'2019'" -> 2019).
brfss_df['Year'] = ( brfss_df['IYEAR']
                     .astype(str)
                     .str.replace("b'", "", regex=False)
                     .str.replace("'", "", regex=False)
                     .astype(int) )
# Rename FMONTH to Month.
brfss_df.rename(columns={'FMONTH': 'Month'}, inplace=True)

# For depression, only use valid responses for ADDEPEV3 (1 for yes, 2 for no).
brfss_valid = brfss_df[brfss_df['ADDEPEV3'].isin([1.0, 2.0])].copy()

# Recode ADDEPEV3: 1 (yes) becomes 1 and 2 (no) becomes 0.
brfss_valid['Depression_binary'] = brfss_valid['ADDEPEV3'].apply(lambda x: 1 if x == 1.0 else 0)

# Group by State, Year, Month to compute the percentage of "yes" responses.
brfss_dep_group = brfss_valid.groupby(['State', 'Year', 'Month']).agg(
    total_valid = ('ADDEPEV3', 'count'),
    yes_count   = ('Depression_binary', 'sum')
).reset_index()

# Calculate the depression rate as a fraction (e.g., 0.2 for 20%).
brfss_dep_group['BRFSS_Depression_Rate'] = brfss_dep_group['yes_count'] / brfss_dep_group['total_valid']
# Convert to percentage (scale 0–100).
brfss_dep_group['BRFSS_Depression_Value'] = brfss_dep_group['BRFSS_Depression_Rate'] * 100

##############################################
# STEP 3: Combine Depression Data with Weighting
##############################################

# Merge the CDC and BRFSS depression data on State, Year, and Month.
dep_merged = pd.merge(
    cdc_dep_grouped,
    brfss_dep_group[['State', 'Year', 'Month', 'BRFSS_Depression_Value']],
    on=['State', 'Year', 'Month'],
    how='outer'
)

# Compute the weighted combined depression value.
# Weighting: 20% from CDC and 80% from BRFSS.
# If one source is missing, use the available value.
def weighted_depression(row):
    # Both values available:
    if not pd.isna(row['CDC_Depression_Value']) and not pd.isna(row['BRFSS_Depression_Value']):
        return 0.2 * row['CDC_Depression_Value'] + 0.8 * row['BRFSS_Depression_Value']
    # Only CDC available:
    elif not pd.isna(row['CDC_Depression_Value']):
        return row['CDC_Depression_Value']
    # Only BRFSS available:
    elif not pd.isna(row['BRFSS_Depression_Value']):
        return row['BRFSS_Depression_Value']
    # Neither available:
    else:
        return np.nan

dep_merged['Combined_Value'] = dep_merged.apply(weighted_depression, axis=1)
# Set the indicator for these rows.
dep_merged['Indicated'] = "Depression"
# Keep only the needed columns.
dep_final = dep_merged[['State', 'Year', 'Month', 'Indicated', 'Combined_Value']]

##############################################
# STEP 4: Prepare Anxiety Data (Using CDC Only)
##############################################

# For anxiety, we use the CDC grouping which already holds the average value.
cdc_anx_grouped['Indicated'] = "Anxiety"
# Rename the CDC value to "Combined_Value" to match the depression file.
cdc_anx_grouped.rename(columns={'CDC_Anxiety_Value': 'Combined_Value'}, inplace=True)
anx_final = cdc_anx_grouped[['State', 'Year', 'Month', 'Indicated', 'Combined_Value']]

##############################################
# STEP 5: Combine Depression and Anxiety, Save File
##############################################

combined = pd.concat([dep_final, anx_final], ignore_index=True)
# Optional: sort the data
combined = combined.sort_values(by=['State', 'Year', 'Month', 'Indicated']).reset_index(drop=True)

# Save the final combined dataset.
combined.to_csv("combined_depression_anxiety.csv", index=False)
print("Combined dataset saved as 'combined_depression_anxiety.csv'.")