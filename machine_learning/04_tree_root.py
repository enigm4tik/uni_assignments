# Machine Learning Basics
# Assignment #04: Tree Root

import pandas as pd
from math import pow
from sklearn import model_selection

# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def find_tree_root(dataframe, columns, target):
    """
    Find the column for the first split
    :param dataframe: the pandas dataframe
    :param columns: categorical column names
    :target: target column name
    :return: dictionary of weighted gini indexes
    """
    weighted_ginis = {column: 0 for column in columns}
    for column in columns:
        weighted_ginis[column] = calculate_weighted_gini_index(dataframe, column, target)
    return weighted_ginis


def calculate_weighted_gini_index(dataframe, column, target):
    """
    Calculate weighted gini index for a column
    :param dataframe: the pandas dataframe
    :param column: categorical column name
    :target: target column name
    :return: dictionary of gini_index for a column
    """
  
    category1, category2 = dataframe[column].unique()
    gini_dictionary = {
        category1: {
            "pass": 0,
            "fail": 0,
        },
        category2: {
            "pass": 0,
            "fail": 0
        }
    }

    cat1 = dataframe[dataframe[column] == category1]
    cat2 = dataframe[dataframe[column] == category2]

    gini_dictionary = calculate_gini_for_leaf(category1, cat1, target,
                                        gini_dictionary)
    gini_dictionary = calculate_gini_for_leaf(category2, cat2, target,
                                        gini_dictionary)
    calculated_index = calculate_weighted_gini(gini_dictionary)
    return calculated_index


def calculate_gini_for_leaf(category, dataframe, target, gini_dictionary):
    """
    Calculate the gini weight for each leaf
    :param category: category to split into
    :param dataframe: the split dataframe
    :param target: target column name
    :param gini_dictionary: dictionary of pass vs. fail 
    :return: updated dictionary
    """
    for pass_or_fail in ["pass", "fail"]:
        gini_dictionary[category][pass_or_fail] = len(
            dataframe[dataframe[target] == pass_or_fail])
    return gini_dictionary


def calculate_weighted_gini(gini_dictionary):
    """
    Calculate the weighted gini index for a categorical variable.
    :param gini_dictionary: 
    { 
        decision1: {"pass": integer, "fail": integer},
        decision2: {"pass": integer, "fail": integer}
    }
    :return: calculated weighted average of gini impurities
    """
    category1, category2 = gini_dictionary.keys()
    leaf_weights = {category: 0 for category in gini_dictionary.keys()}
    sums = {category: 0 for category in gini_dictionary.keys()}
    for category, pass_or_fail in gini_dictionary.items():
        pass_count = pass_or_fail["pass"]
        fail_count = pass_or_fail["fail"]
        sum_count = pass_count + fail_count
        sums[category] = sum_count
        leaf_weights[category] = 1 - pow(pass_count / sum_count, 2) - pow(
            fail_count / sum_count, 2)
        leaf_weights[category] = round(leaf_weights[category], 2)

    sum_of_sums = sums[category1] + sums[category2]
    weighted_average_of_gini_impurities = 0
    for category in gini_dictionary.keys():
        weighted_average_of_gini_impurities += sums[
            category] / sum_of_sums * leaf_weights[category]

    return weighted_average_of_gini_impurities

def pretty_print_results(dataframe, columns, target, with_table=False):
    """
    Print the results in a pretty way. 
    :param dataset: pandas dataframe
    :param columns: columns to consider
    :param target: target column name
    :param with_table: adds table of all results
    """
    all_ginis = find_tree_root(dataframe, columns, target)
    print(f"\nRoot split at: {min(all_ginis, key=all_ginis.get)}\n")
    if with_table: 
      print("All weighted gini scores:\n")
      print(f"{'Column':12} | Weighted Gini Index")
      print(f"{'':_^36}")
      for key, value in all_ginis.items():
          print(f"{key:12} | {round(value, 4)}")
      print(f"{'':_^36}")


## DATASET DESCRIPTION
# This is a dataset based on school children and their performance in class 1 through 3. 
# A lot of categorical variables have been collected (eg. living in urban or rural areas).
# More information about this dataset: https://www.kaggle.com/datasets/whenamancodes/student-performance
# This dataset is consisting of 394 rows and 33 columns of which 12 are categorical.
# The target is chosen as G3. 

math_students = pd.read_csv("Maths.csv")
target = "G3"
interesting_columns = [
    "Pstatus", "schoolsup", "paid", "higher", "activities", "nursery",
    "internet", "sex", "school", "address", "famsize", "romantic"
]

# Create a new DataFrame with only the interesting columns and the target

math = pd.DataFrame(math_students, columns=[*interesting_columns, target])

# Manipulate G3 to show classes "pass" or "fail" based on their grades
# 0-10 fail
# 11-20 pass

rows = len(math)

for i in range(rows):
    math.at[i, target] = "pass" if math.at[i, target] > 10 else "fail"

# Split into test and train data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    math.iloc[:, : 29], math[target], shuffle=True, train_size=0.8)

# Figure out the split for train and test data (use True to see all values)
pretty_print_results(X_train, interesting_columns, target, with_table=True)
pretty_print_results(X_test, interesting_columns, target, with_table=True)

# I realized that in my dataset, the gini impurity indexes are VERY close.
# It makes sense that for most runs the two root splits will be different for test and training data. 
# Decision trees tend to overfit and changing just a little bit of data between training and test set 
# can greatly change the decision.
