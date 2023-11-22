# Machine Learning Basics
# Assignment #03: Data Handling

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer

# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def find_inappropriate_value_in_column(dataframe, column, datatype):
    """
  Iterate over dataframe and figure out if datapoints are 
  inappropriate for their datatype
  :param dataframe: pandas dataframe
  :param column: column name
  :param datatype: the expected datatype
  :return: rows with wrong datapoints for datatypes, bad_values
  """

    rows_that_need_work = []
    bad_values = []
    if not datatype == "object":
        for value in wine[column]:
            if value:
                try:
                    float(value)
                except ValueError:
                    if not value in bad_values:
                        bad_values.append(value)
                        row_that_needs_work = wine[column].loc[
                            lambda x: x == value].index[0]
                        if not row_that_needs_work in rows_that_need_work:
                            rows_that_need_work.append(row_that_needs_work)
    return rows_that_need_work, bad_values


def find_problematic_rows_and_values(dataframe, data_types):
    """
  Find problematic rows and values in a dataframe based on datatype
  :param dataframe: pandas dataframe
  :param data_types: dictionary {coluimn_name: data_type}
  :return: all problematic roles and values
  """
    problematic_rows = []
    problematic_values = []
    for column, datatype in data_types.items():
        bad_rows, bad_values = find_inappropriate_value_in_column(
            wine, column, datatype)
        for bad_row in bad_rows:
            if not bad_row in problematic_rows:
                problematic_rows.append(bad_row)
        for bad_value in bad_values:
            if not bad_value in problematic_values:
                problematic_values.append(bad_value)
    return problematic_rows, problematic_values


def apply_column_datatype_to_columns(dataframe, data_types):
    """
  Apply datatype to column if possible
  If not, print the corrupt column.
  :param dataframe: pandas dataframe
  :param data_types: datatype dictionary {column_name: data_type}
  :return: updated data_frame
  """
    for column in dataframe.columns:
        try:
            wine[column] = wine[column].astype(data_types[column])
        except ValueError:
            print(f"This column is corrupt: {column}")
    return dataframe


def replace_nan_values_with_nan(dataframe, nan_values):
    """
  Remove missing values from dataframe
  :param dataframe: pandas dataframe
  :param nan_values: a list of nan_values
  :return: updated dataframe
  """
    for nan_value in nan_values:
        dataframe = dataframe.replace(nan_value, np.nan)
    return dataframe


# Define data types for all columns
column_data_types = {
    "alcohol": "float64",
    "malic_acid": "float64",
    "ash": "float64",
    "alcalinity_of_ash": "float64",
    "magnesium": "float64",
    "total_phenols": "float64",
    "flavanoids": "float64",
    "nonflavanoid_phenols": "float64",
    "proanthocyanins": "float64",
    "color_intensity": "float64",
    "hue": "float64",
    "od280/od315_of_diluted_wines": "float64",
    "proline": "float64",
    "color": "int",
    "season": "object",
    "target": "int",
    "country-age": "object",
}

## STEP 1 ------------------------------------------------------------------------
## Read the data
# Choose ; as the separator, and indicate the header in row 1 and skip the footer.
# Bad Lines are skipped.
wine = pd.read_csv("assignment_datahandling_wine_exercise.csv",
                   on_bad_lines='skip',
                   skiprows=0,
                   skipfooter=1,
                   sep=';',
                   header=1,
                   engine='python')

## STEP 2 ------------------------------------------------------------------------
## Replace nan values
# Get problematic rows and values using definded functions
problematic_rows, problematic_values = find_problematic_rows_and_values(
    wine, column_data_types)

# Define the nan values that need to be replaced with nan
# Replace the nan values
nan_values = ["-999", float(-999), "missing", -999]
wine = replace_nan_values_with_nan(wine, nan_values)

## STEP 3 ------------------------------------------------------------------------
## Fix individual rows: 50, 51 and 142
# Fix row with index 50

all_values_in_first_row = wine.iloc[50][0].split(",")
for index, column in enumerate(wine.columns):
    wine.at[50, column] = all_values_in_first_row[index]

# Fix the row with index 51
for column in wine.columns:
    try:
        my_string = wine.at[51, column].replace(",", ".")
        my_float = round(float(my_string), 2)
        wine.at[51, column] = my_float
    except (AttributeError, ValueError):
        continue

# Fix the row with index 142
correct_values = wine.iloc[142][0].split(",")

for column in wine.columns:
    if column == 'alcohol':
        continue
    current_value = wine.at[142, column]
    if not current_value in ["nan", None]:
        correct_values.append(current_value)

for index, column in enumerate(wine.columns):
    wine.at[142, column] = correct_values[index]

## STEP 4 ------------------------------------------------------------------------
## Apply the correct datatypes to the values
print("\n=== Corrupt columns ===\n")
wine = apply_column_datatype_to_columns(wine, column_data_types)
# Column magnesium is corrupt

# Fix column magnesium:
column_to_fix = "magnesium"
# print(wine[column_to_fix]) # row 166 has the problem
# print(wine.iloc[166]) # row 166 has 111 and 1.7 but no value for total_phenols

# Warning: In this case I assume that 111 is the value for magnesium while 1.7 is the
# value for total_phenols. The best case would be to ask the data provider.

# Fix row 166:
double_value = wine.at[166, column_to_fix].split(" ")
wine.at[166, column_to_fix] = double_value[0]
wine.at[166, "total_phenols"] = double_value[1]

wine[column_to_fix] = wine[column_to_fix].astype("float")
wine["total_phenols"] = wine["total_phenols"].astype("float")

## STEP 5 ------------------------------------------------------------------------
## Replace nan values again, because there was -999 in one of the corrupt samples.
wine = replace_nan_values_with_nan(wine, nan_values)

## STEP 6 ------------------------------------------------------------------------
## Separate numerical and categorical data

categorics = []
numericals = []
for column, datatype in column_data_types.items():
    if datatype == "object":
        categorics.append(column)
    elif datatype != "int":
        numericals.append(column)

## STEP 7 ------------------------------------------------------------------------
## Fix the season column
print("\n=== Categorical variables ===\n")
for categorical_variable in categorics:
    print(wine[categorical_variable].value_counts())

# We see that "spring" and "aut" are used instead of their category - replace
wine = wine.replace("spring", "SPRING")
wine = wine.replace("aut", "AUTUMN")

## STEP 8 ------------------------------------------------------------------------
## Split country and age that are stored in one column
wine[['country', 'age']] = wine['country-age'].str.split('-', expand=True)
wine = wine.drop(columns="country-age")
column_data_types["country"] = "object"

# I decided to remove "years" from age and use int as its datatype
column_data_types["age"] = "int"
wine["age"] = wine["age"].str.replace("years", "")

# Adjust datatypes again
wine = apply_column_datatype_to_columns(wine, column_data_types)

## STEP 9 ------------------------------------------------------------------------
## Exclude duplicates
wine.duplicated(keep="last")

## STEP 10 -----------------------------------------------------------------------
## Find Outliers
# Create a box-whiskers plot for each column that is not categorical
print("\n=== Outliers ===\n")

# Comment out this block for plots!

# for index, column in enumerate(wine.columns):
#     if column_data_types[column] not in ["object", "int"]:
#         plt.figure(index)
#         plt.title(column.upper())
#         plt.boxplot(wine[column])

# End Comment

# Outliers: It would be best to talk to some wine specialist to figure out if the values are within range.

# Visually the following columns have possible outliers:
# Alcalinity, Magnesium, Total Phenols, Proanthocyanine, Color Intensity, Hue

outlier_columns = wine[[
    "alcalinity_of_ash", "magnesium", "total_phenols", "proanthocyanins",
    "color_intensity", "hue"
]]

print(outlier_columns.describe())

# Looking at the values (without knowledge!) it's difficult to decide.
# The total phenols look like actual outliers so for the sake of this assignment, I will remove these.
print("\n=== 5 largest total phenol rows ===\n")
print(outlier_columns.nlargest(5, "total_phenols"))
# It's sample 167 and 168
wine = wine.drop([167, 168])
print("\n=== Wine after dropping two outliers ===\n")
print(wine.describe())

## STEP 11 -----------------------------------------------------------------------
## Impute missing data using KNNImputer on numerical data

knn_imputer = KNNImputer(n_neighbors=5)
numerical_data = wine[numericals]
imputed_wine = pd.DataFrame(knn_imputer.fit_transform(numerical_data),
                            columns=numerical_data.columns,
                            index=numerical_data.index)
wine.update(imputed_wine)

# All missing data is now imputed
print("\n=== Wine after imputation ===\n")
print(wine.describe())

## STEP 11 -----------------------------------------------------------------------
## Get class distribution of the target variable
wine_grouped = wine[[*numericals, "target"]].groupby("target")

## Comment in the following for loop to see the distribution
# of the target value for all numerical data

# for index, column in enumerate(numericals):
#   plt.figure(index+20)
#   plt.title(column.upper())
#   wine[wine["target"]==0][column].plot.kde(label="Target 0")
#   wine[wine["target"]==1][column].plot.kde(label="Target 1")
#   wine[wine["target"]==2][column].plot.kde(label="Target 2")
#   plt.legend()
# plt.show()

# End Comment

## STEP 12 -----------------------------------------------------------------------
## Group magnesium by color and calculate statistics within groups
print("\n=== Magnesium grouped by color ===\n")
magnesium_group = wine[["magnesium", "color"]].groupby("color")
print(magnesium_group.describe())
