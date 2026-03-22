REQUIRED = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "median_house_value"
    ]

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:  
    """
    Reads data from a CSV file and creates a DataFrame.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}") #Check for required columns
    return df

df = load_data("data\housing.csv")

def check_data(df: pd.DataFrame) -> None:
    """Function to check data quality."""
    print(df.info())  # Check data types
    print("\n===========================================================================\n")
    print("Number of rows and columns:", df.shape)  # Check shape of the dataframe
    print("\nMissing values per column:\n", df.isna().sum())  # Check for missing values
    print("\nNumber of duplicate rows:", df.duplicated().sum())  # Check for duplicates
    print("\n===========================================================================\n")
    print("First 5 rows of the dataframe:\n", df.head())  # Preview first 5 rows

def stats_data(df: pd.DataFrame) -> None:

    print("Statistical summary:\n", df.describe())  # Summary statistics