import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PART 1: PROCESS AND MERGE DATA
# -------------------------------

############# CDC Processing #############
# Load CDC data (assumes file "cdc_reduced.csv")
cdc_df = pd.read_csv("cdc_reduced.csv")

# Convert the "Time Period Start Date" to datetime; extract Year and Month
cdc_df['StartDate'] = pd.to_datetime(cdc_df['Time Period Start Date'], errors='coerce')
cdc_df['Year'] = cdc_df['StartDate'].dt.year
cdc_df['Month'] = cdc_df['StartDate'].dt.month

# For depression we use rows that have an indicator for depression.
# We consider both "Symptoms of Depressive Disorder" and
# "Symptoms of Anxiety Disorder or Depressive Disorder" for depression.
cdc_dep_indicators = ["Symptoms of Depressive Disorder", "Symptoms of Anxiety Disorder or Depressive Disorder"]
cdc_dep = cdc_df[ cdc_df['Indicator'].isin(cdc_dep_indicators) ].copy()

# Group by State, Year, and Month and average the Value.
cdc_dep_grouped = (
    cdc_dep.groupby(['State', 'Year', 'Month'], as_index=False)
           .agg({'Value': 'mean'})
)
cdc_dep_grouped.rename(columns={'Value': 'CDC_Depression_Value'}, inplace=True)

############# BRFSS Processing #############
# Load BRFSS data (assumes file "brfss_reduced.csv")
brfss_df = pd.read_csv("brfss_reduced.csv")

# Clean the IYEAR column: Remove characters like "b'" and "'" (e.g., turn "b'2019'" into 2019)
brfss_df['Year'] = ( brfss_df['IYEAR']
                    .astype(str)
                    .str.replace("b'", "", regex=False)
                    .str.replace("'", "", regex=False)
                    .astype(int) )
# Rename FMONTH to Month
brfss_df.rename(columns={'FMONTH': 'Month'}, inplace=True)

# For depression, only keep valid responses on ADDEPEV3 (1 for yes and 2 for no)
brfss_valid = brfss_df[ brfss_df['ADDEPEV3'].isin([1.0, 2.0]) ].copy()

# Create a binary column: 1 if response equals 1 (yes), and 0 if response equals 2 (no)
brfss_valid['Depression_binary'] = brfss_valid['ADDEPEV3'].apply(lambda x: 1 if x==1.0 else 0)

# Group by State, Year, Month, count how many valid responses and sum the yes responses.
brfss_dep_group = brfss_valid.groupby(['State', 'Year', 'Month']).agg(
    total_valid = ('ADDEPEV3', 'count'),
    yes_count   = ('Depression_binary', 'sum')
).reset_index()

# Calculate the depression rate as the fraction of yes responses.
brfss_dep_group['BRFSS_Depression_Rate'] = brfss_dep_group['yes_count'] / brfss_dep_group['total_valid']
# Multiply by 100 so that we are on a percentage scale (to compare with CDC).
brfss_dep_group['BRFSS_Depression_Value'] = brfss_dep_group['BRFSS_Depression_Rate'] * 100

############# Merge CDC and BRFSS for Depression #############
# Merge the two dataframes on State, Year, Month (use outer join so we keep all available data)
dep_merged = pd.merge(
    cdc_dep_grouped, 
    brfss_dep_group[['State', 'Year', 'Month', 'BRFSS_Depression_Value']],
    on=['State', 'Year', 'Month'],
    how='outer'
)

# Create a combined depression value from both sources:
def combine_values(row):
    vals = []
    if not pd.isna(row['CDC_Depression_Value']):
        vals.append(row['CDC_Depression_Value'])
    if not pd.isna(row['BRFSS_Depression_Value']):
        vals.append(row['BRFSS_Depression_Value'])
    if len(vals) > 0:
        return np.mean(vals)
    return np.nan

dep_merged['Combined_Depression_Value'] = dep_merged.apply(combine_values, axis=1)

# For validity and debugging, keep both individual columns as well as the combined column.
dep_merged['Indicated'] = "Depression"
# Reorder columns for clarity:
dep_merged = dep_merged[['State', 'Year', 'Month', 'Indicated',
                           'CDC_Depression_Value', 'BRFSS_Depression_Value',
                           'Combined_Depression_Value']]

# Save the merged validation file so you can inspect both values.
dep_merged.to_csv("combined_depression_validation.csv", index=False)
print("Saved merged depression validation file as 'combined_depression_validation.csv'.")

# -------------------------------
# PART 2: EDA on Depression Trends
# -------------------------------
# We want to show trends over past years for both CDC and BRFSS.

# For the purpose of EDA, we aggregate over all states.
# Group by Year and compute the average of each source's values.
trend_agg = dep_merged.groupby('Year').agg({
    'CDC_Depression_Value': 'mean',
    'BRFSS_Depression_Value': 'mean',
    'Combined_Depression_Value': 'mean'
}).reset_index()

# Save this aggregated trend as a CSV so you can check the numbers.
trend_agg.to_csv("depression_trend_by_year.csv", index=False)
print("Saved aggregated depression trend data as 'depression_trend_by_year.csv'.")

# Create a line plot of the depression trends (averaged across all states) over the years.
plt.figure(figsize=(10, 6))
sns.lineplot(data=trend_agg, x='Year', y='CDC_Depression_Value',
             marker='o', label='CDC Depression')
sns.lineplot(data=trend_agg, x='Year', y='BRFSS_Depression_Value',
             marker='s', label='BRFSS Depression')
sns.lineplot(data=trend_agg, x='Year', y='Combined_Depression_Value',
             marker='^', label='Combined Depression')
plt.title("Average Depression Trend by Year (All States)")
plt.ylabel("Depression Value (%)")
plt.xlabel("Year")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("depression_trend_over_years.png")
plt.show()
print("Saved depression trend plot as 'depression_trend_over_years.png'.")