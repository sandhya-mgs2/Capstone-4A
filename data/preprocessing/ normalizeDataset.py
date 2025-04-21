import pandas as pd

# --- Define the Columns to Extract ---

# For BRFSS, select columns capturing survey metadata and mental health indicators.
brfss_cols = [
    '_STATE',    # State identifier (numeric code)
    'FMONTH',    # Survey month
    'IYEAR',     # Survey year (currently stored as bytes, e.g., b'2019')
    'MENTHLTH',  # Number of days mental health was not good
    'ADDEPEV3',  # Indicator for depression diagnosis
    'CDWORRY',   # Possibly related to anxiety (e.g., excessive worry)
]

# For CDC, include the "Group" column so we can filter "By State" rows.
cdc_cols = [
    'Indicator',
    'Group',
    'State',
    'Time Period Start Date',
    'Time Period End Date',
    'Value',
    'Low CI',
    'High CI'
]

# --- Load the CDC Dataset ---

# Load CDC data with the selected columns.
cdc_df = pd.read_csv("cdc.csv", usecols=cdc_cols)

# Filter to include only rows where 'Group' is "By State"
cdc_df = cdc_df[cdc_df['Group'] == 'By State']

# --- Load the BRFSS Dataset ---

# Load BRFSS data using only the required columns to save memory.
brfss_df = pd.read_csv("brfss.csv", usecols=brfss_cols)

# Rename _STATE to State for consistency.
brfss_df.rename(columns={'_STATE': 'State'}, inplace=True)

# Decode IYEAR column from bytes to a standard string if needed.
brfss_df['IYEAR'] = brfss_df['IYEAR'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# --- Map Numeric State Codes to State Names ---

# Define the state mapping dictionary based on your provided mapping.
state_map = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
    66: "Guam",
    72: "Puerto Rico",
    78: "Virgin Islands"
}

# Convert the numeric state code to an integer and map it to state names.
brfss_df['State'] = brfss_df['State'].astype(int).map(state_map)

# --- Save the Reduced Datasets as Separate CSV Files ---

cdc_df.to_csv("cdc_reduced.csv", index=False)
brfss_df.to_csv("brfss_reduced.csv", index=False)

print("Reduced datasets saved as 'cdc_reduced.csv' and 'brfss_reduced.csv'")
