import pandas as pd
import re

def normalize_column(col):
    """
    Remove trailing suffixes (like '.1', '.2', etc.) from a column name.
    """
    return re.sub(r'\.\d+$', '', col)

def fix_csv(input_file, output_file):
    # Read the CSV; assuming the first column is 'date' and set as index.
    df = pd.read_csv(input_file, index_col=0)
    
    # Create a mapping from normalized column name to the list of columns that match
    column_groups = {}
    for col in df.columns:
        base = normalize_column(col)
        column_groups.setdefault(base, []).append(col)
    
    # Create a new DataFrame to hold the fixed columns
    fixed_df = pd.DataFrame(index=df.index)
    
    # For each group of columns with the same normalized name,
    # take the first non-null value across the duplicates for each row.
    for base, cols in column_groups.items():
        # bfill along columns will fill NaNs with the next valid value in that row.
        fixed_df[base] = df[cols].bfill(axis=1).iloc[:, 0]
    
    # (Optional) Sort columns alphabetically if desired.
    fixed_df = fixed_df.sort_index(axis=1)
    
    # Save the fixed DataFrame to a new CSV file.
    fixed_df.to_csv(output_file)
    print(f"Fixed CSV saved to: {output_file}")

if __name__ == '__main__':
    input_csv = "us_trends_weekly.csv"         # your original file
    output_csv = "us_trends_weekly_fixed.csv"    # the fixed output file
    fix_csv(input_csv, output_csv)
