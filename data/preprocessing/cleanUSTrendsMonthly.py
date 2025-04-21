import pandas as pd
import numpy as np

def non_zero_mean(series):
    """
    Computes the mean of a pandas Series ignoring zeros and NaNs.
    If there are no valid (non-zero) values, returns 0.
    """
    valid = series[(series != 0) & (series.notna())]
    if valid.empty:
        return 0
    else:
        return valid.mean()

# Read the fixed dataset, assuming the date column is the index and parse as datetime.
df = pd.read_csv("us_trends_weekly_fixed.csv", index_col=0, parse_dates=True)

# Ensure that empty strings (if any) are treated as NaN.
df.replace("", np.nan, inplace=True)

# Group by month using pd.Grouper with a monthly frequency, then apply our custom aggregation.
monthly_avg = df.groupby(pd.Grouper(freq='M')).agg(non_zero_mean)

# Reformat the index to display only "year-month"
monthly_avg.index = monthly_avg.index.strftime("%Y-%m")

# Round values to 2 decimal places.
monthly_avg = monthly_avg.round(2)

# Save the monthly averages to a CSV file.
monthly_avg.to_csv("us_trends_monthly.csv")
print("Monthly averages saved to us_trends_monthly.csv")

# Identify columns that have no 0 values in the final output.
columns_no_zero = [col for col in monthly_avg.columns if (monthly_avg[col] == 0).sum() == 0]
print("Columns with no 0 values:", columns_no_zero)

# Create a cleaned DataFrame that contains only columns with no 0 values.
monthly_avg_cleaned = monthly_avg[columns_no_zero]

# Save the cleaned DataFrame to a CSV file.
monthly_avg_cleaned.to_csv("us_trends_monthly_cleaned.csv")
print("Cleaned monthly averages (only columns with no 0 values) saved to us_trends_monthly_cleaned.csv")
