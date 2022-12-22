import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from random import sample
import re
from nltk import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import inspect
from typing import List, Dict, Optional, Tuple

def split_column(df, column, delimiter, new_columns):
    """Split the values in the specified column of the DataFrame into multiple columns based on the specified delimiter."""
    df[new_columns] = df[column].str.split(delimiter, expand=True)
    return df

def merge_columns(df, columns, delimiter, new_column):
    """Merge the values in the specified columns of the DataFrame into a single column, with the values from each column separated by the specified delimiter."""
    df[new_column] = df[columns].apply(lambda x: delimiter.join(x), axis=1)
    return df

def aggregate_by_group(df, column, aggregation):
    """Group the rows of the DataFrame by the values in the specified column and apply the specified aggregation function to each group."""
    df = df.groupby(column).agg(aggregation)
    return df


def add_date_offset(df, column, offset, unit):
    """Add the specified number of days, weeks, months, or years to the date column."""
    if unit == "days":
        df[column] = df[column] + pd.DateOffset(days=offset)
    elif unit == "weeks":
        df[column] = df[column] + pd.DateOffset(weeks=offset)
    elif unit == "months":
        df[column] = df[column] + pd.DateOffset(months=offset)
    elif unit == "years":
        df[column] = df[column] + pd.DateOffset(years=offset)
    return df

def add_time_offset(df, column, offset, unit):
    """Add the specified number of seconds, minutes, hours, or days to the time column."""
    if unit == "seconds":
        df[column] = df[column] + pd.Timedelta(seconds=offset)
    elif unit == "minutes":
        df[column] = df[column] + pd.Timedelta(minutes=offset)
    elif unit == "hours":
        df[column] = df[column] + pd.Timedelta(hours=offset)
    elif unit == "days":
        df[column] = df[column] + pd.Timedelta(days=offset)
    return df


def convert_to_datetime(df, column):
    """Convert the values in the specified column of the DataFrame to the datetime data type."""
    df[column] = pd.to_datetime(df[column])
    return df

def extract_date_parts(df, column, parts):
    """Extract the specified parts of the date, such as the year, month, and day, and create new columns for each part."""
    df[parts] = df[column].dt.strftime("%Y %m %d").str.split(" ", expand=True)
    return df

def extract_time_parts(df, column, parts):
    """Extract the specified parts of the time, such as the hour, minute, and second, and create new columns for each part."""
    df[parts] = df[column].dt.strftime("%H %M %S").str.split(" ", expand=True)
    return df


def remove_numbers(df, column):
    """Remove numerical values from the specified column of the DataFrame."""
    df[column] = df[column].str.replace(r"\d+", "")
    return df

def remove_white_spaces(df, column):
    """Remove extra white space, such as leading and trailing spaces, from the values in the specified column of the DataFrame."""
    df[column] = df[column].str.strip()
    return df

def remove_short_words(df, column, min_length):
    """Remove words that are shorter than the specified length from the specified column of the DataFrame."""
    df[column] = df[column].apply(lambda x: " ".join([word for word in x.split() if len(word) >= min_length]))
    return df

def remove_short_chars(df, columns, min_length):
    """Remove rows from the DataFrame that contain characters shorter than the specified length in all of the specified columns."""
    df = df[df[columns].apply(lambda x: all(len(x[col]) >= min_length for col in columns), axis=1)]
    return df


def remove_long_chars(df, columns, max_length):
    """Remove rows from the DataFrame that contain characters longer than the specified length in all of the specified columns."""
    df = df[df[columns].apply(lambda x: all(len(x[col]) <= max_length for col in columns), axis=1)]
    return df


def remove_long_words(df, column, max_length):
    """Remove words that are longer than the specified length from the specified column of the DataFrame."""
    df[column] = df[column].apply(lambda x: " ".join([word for word in x.split() if len(word) <= max_length]))
    return df

def remove_capitalized_words(df, column):
    """Remove words that are capitalized from the specified column of the DataFrame."""
    df[column] = df[column].apply(lambda x: " ".join([word for word in x.split() if not word.isupper()]))
    return df

def lemmatize_values(df, column):
    """Lemmatize the values in the specified column of the DataFrame, reducing inflected or derived words to their word stem or base form."""
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df

def stem_values(df, column):
    """Stem the values in the specified column of the DataFrame, reducing words to their word stem or base form."""
    stemmer = PorterStemmer()
    df[column] = df[column].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    return df

def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Count the number of duplicate rows in a Pandas DataFrame."""
    return df.duplicated().sum()

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame and return the modified DataFrame."""
    return df.drop_duplicates()

def replace_missing_values(df, default_values):
    """Replace missing values in the DataFrame with default values and return the modified DataFrame."""
    return df.replace(pd.np.nan, default_values)

def replace_null_values_pair(df, value_pairs_list):
    for column_value_pairs in value_pairs_list:  
        for column, value in column_value_pairs:
            df[column] = df[column].fillna(value)
    return df


def convert_data_types(df, data_types):
    """Convert the data types of columns in the DataFrame and return the modified DataFrame."""
    return df.astype(data_types)

def remove_specific_values(df, column, values):
    """Remove rows from the DataFrame that have specific values in the specified column."""
    return df[~df[column].isin(values)]

def add_calculated_column(df, column_name, calculation):
    """Add a new column to the DataFrame that is the result of a calculation using other columns."""
    df[column_name] = calculation
    return df

def reorder_columns(df, column_order):
    """Reorder the columns in the DataFrame according to the specified column order."""
    return df[column_order]

def rename_columns(df, column_names):
    """Rename the columns in the DataFrame according to the specified mapping of old and new names."""
    return df.rename(columns=column_names)

def filter_rows(df, column, min_value, max_value):
    """Filter the DataFrame to only include rows that have values in the specified column within a certain range."""
    return df.loc[df[column].between(min_value, max_value)]


def remove_outliers(df, column, min_value, max_value):
    """Remove rows from the DataFrame that have values in the specified column outside of a certain range."""
    return df.loc[df[column].between(min_value, max_value)]

def normalize_values(df, column):
    """Normalize the values in the specified column of the DataFrame by scaling them to have a mean of 0 and a standard deviation of 1."""
    return (df[column] - df[column].mean()) / df[column].std()

def aggregate_values(df, group_column, value_column, aggregation_function):
    """Group the rows in the DataFrame by the values in the specified group column and apply the specified aggregation function to the values in the specified value column."""
    return df.groupby(group_column)[value_column].aggregate(aggregation_function)

def sort_values(df, column, ascending=True):
    """Sort the rows in the DataFrame by the values in the specified column."""
    return df.sort_values(by=column, ascending=ascending)

def remove_columns(df, columns):
    """Remove the specified columns from the DataFrame."""
    return df.drop(columns, axis=1)


def remove_empty_rows(df):
    """Remove rows from the DataFrame that have no values in any of the columns."""
    return df.dropna(how="all")

def remove_empty_columns(df):
    """Remove columns from the DataFrame that have no values in any of the rows."""
    return df.dropna(how="all", axis=1)

def remove_duplicate_columns(df):
    """Remove duplicate columns from the DataFrame, where two columns are considered to be duplicates if they have the same values in all of the rows."""
    return df.T.drop_duplicates().T

def encode_categorical_values(df, columns, encoding_scheme):
    """Encode the categorical values in the specified columns of the DataFrame using the specified encoding scheme."""
    if encoding_scheme == "one-hot":
        df = pd.get_dummies(df, columns=columns)
    elif encoding_scheme == "ordinal":
        for column in columns:
            df[column] = df[column].astype("category").cat.codes
    return df

def remove_irrelevant_columns(df, columns):
    """Remove the specified columns from the DataFrame if they are not relevant to the specific analysis that is being performed."""
    return df.drop(columns, axis=1)

def remove_duplicate_values(df, column):
    """Remove duplicate values from the specified column of the DataFrame."""
    return df[column].drop_duplicates()

def remove_special_characters(df, column):
    """Remove special characters, such as punctuation marks and non-alphanumeric characters, from the values in the specified column of the DataFrame."""
    df[column] = df[column].str.replace(r"[^a-zA-Z0-9]", "")
    return df

def remove_stopwords(df, column, stopwords):
    """Remove the specified stopwords from the values in the specified column of the DataFrame."""
    df[column] = df[column].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))
    return df

def clean_text(df, column, min_word_length=None, max_word_length=None):
    """Clean the text in the specified column of the DataFrame by removing numbers, white space, short words, special characters and long words."""
    # Remove numerical values from the column
    df = remove_numbers(df, column)
    
    # Remove punctuation
    df[column] = df[column].str.replace(r"[^\w\s]", "")
    # Convert to lowercase
    df[column] = df[column].str.lower()
    # Remove extra white space, such as leading and trailing spaces, from the column
    df = remove_white_spaces(df, column)

    df = remove_special_characters(df, column)
    # Remove words that are shorter than the specified length from the column
    df = remove_short_chars(df, column, min_word_length)

    # Remove words that are longer than the specified length from the column
    df = remove_long_chars(df, column, max_word_length)

    return df

def clean_text_columns(df, columns, min_word_length=None, max_word_length=None):
    """Clean the text in the specified columns of the DataFrame by removing numbers, white space, short words, special characters and long words."""
    for column in columns:
        # Remove numerical values from the column
        df = remove_numbers(df, column)
        
        # Remove punctuation
        df[column] = df[column].str.replace(r"[^\w\s]", "")
        # Convert to lowercase
        df[column] = df[column].str.lower()
        # Remove extra white space, such as leading and trailing spaces, from the column
        df = remove_white_spaces(df, column)
    
        df = remove_special_characters(df, column)
        # Remove words that are shorter than the specified length from the column
        df = remove_short_chars(df, column, min_word_length)
    
        # Remove words that are longer than the specified length from the column
        df = remove_long_chars(df, column, max_word_length)

    return df


def clean_and_normalize_text_column(df, column, min_word_length, stemming, lemmatization):
    """Clean and normalize the specified text column by removing numerical values, extra white space, short words, and performing stemming and/or lemmatization."""
    df = remove_numbers(df, column)
    df = remove_white_spaces(df, column)
    df = remove_short_words(df, column, min_word_length)
    
    def normalize_text(text):
        if stemming:
            stemmer = PorterStemmer()
            text = " ".join([stemmer.stem(word) for word in text.split()])
        if lemmatization:
            lemmatizer = WordNetLemmatizer()
            text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    df[column] = df[column].apply(normalize_text)
    return df

def convert_and_extract_date(df, column, date_parts):
    df = convert_to_datetime(df, column)
    df = extract_date_parts(df, column, date_parts)
    return df
def convert_and_extract_time(df, column, time_parts):
    df = convert_to_datetime(df, column)
    df = extract_time_parts(df, column, time_parts)
    return df

def clean_text_extra(df, column):
    """Clean the text in the specified column of the DataFrame by removing numbers, punctuation, and stop words, and lemmatizing and stemming the remaining words."""
    # Remove numbers
    df[column] = df[column].str.replace(r"\d+", "")
    # Remove punctuation
    df[column] = df[column].str.replace(r"[^\w\s]", "")
    # Convert to lowercase
    df[column] = df[column].str.lower()
    # Tokenize the words
    df[column] = df[column].apply(word_tokenize)
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    df[column] = df[column].apply(lambda x: [word for word in x if word not in stop_words])
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    # Stem the words
    stemmer = PorterStemmer()
    df[column] = df[column].apply(lambda x: [stemmer.stem(word) for word in x])
    # Remove short words
    df[column] = df[column].apply(lambda x: [word for word in x if len(word) > 2])
    # Join the words back into a single string
    df[column] = df[column].apply(lambda x: " ".join(x))
    return df

def dataframe_size(df):
    """Calculates the size of a pandas DataFrame in megabytes."""
    # Calculate the memory usage of the DataFrame
    memory_usage = df.memory_usage().sum()
    
    # Convert the memory usage to megabytes
    size_in_mb = memory_usage / 1_000_000
    
    return round(size_in_mb,2)

def calculate_dimensions(df):
    # calculate the number of rows and columns in the dataframe
    rows, columns = df.shape
    return rows, columns

def calculate_null_values_column(df):
    # calculate the total number of null values in each column
    null_values = df.isnull().sum()
    return null_values

def calculate_null_values_total(df):
    # calculate the total number of null values in the entire dataframe
    null_values = df.isnull().sum().sum()
    return null_values

def calculate_null_ratio_row(df):
    # calculate the ratio of null values to non-null values in each row
    null_ratio = df.isnull().sum(axis=1) / (df.shape[1] - df.isnull().sum(axis=1))
    return null_ratio

def calculate_null_ratio_total(df):
    # calculate the ratio of null values to non-null values in the dataframe
    null_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1] - df.isnull().sum().sum())
    return null_ratio

def calculate_null_percentage(df):
    # calculate the percentage of null values to non-null values in the dataframe
    null_percentage = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    return round(null_percentage,1)

def highest_null_column(df):
    # Create a new dataframe containing boolean values indicating whether each value is null
    null_df = df.isnull()

    # Sum the number of null values in each column
    null_counts = null_df.sum()

    # Find the column with the highest sum of null values
    highest_null_column = null_counts.idxmax()

    # Get the count of null values in the highest null column
    highest_null_count = null_counts[highest_null_column]

    return highest_null_column, highest_null_count

def find_column_with_most_nulls(df):
    null_counts = df.isnull().sum()
    most_nulls = null_counts.max()
    most_null_column = null_counts[null_counts == most_nulls].index[0]
    return most_null_column, most_nulls

def remove_null_values_column(df, columns):
    # remove null values from the specified column
    for column in columns:
        df = df[pd.notnull(df[column])]
    return df


def remove_null_values_row(dataframe, row_index):
    # Get the row at the specified index
    row = dataframe.iloc[row_index]
    
    # Remove any null values in the row
    row = row.dropna()
    
    # Update the dataframe with the new row
    dataframe.iloc[row_index] = row
    
    # Return the updated dataframe
    return dataframe

def remove_null_values_total(df):
    # remove null values from the dataframe
    df = df.dropna()
    return df

def replace_null_values_column(df, column, value):
    # replace null values in the specified column with the given value
    df[column].fillna(value, inplace=True)
    return df

def replace_null_values(df, value):
    # replace null values in the dataframe with the given value
    df.fillna(value, inplace=True)
    return df

def replace_null_values_mean(df, column):
    # check if the column is numeric
    if df[column].dtype in ["int64", "float64"]:
        # replace null values in the column with the mean of the column
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        # if the column is not numeric, do nothing
        pass
    return df

def replace_null_values_median(df, column):
    # check if the column is numeric
    if df[column].dtype in ["int64", "float64"]:
        # replace null values in the column with the median of the column
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # if the column is not numeric, do nothing
        pass
    return df


def replace_missing_values_with_knn_column(df, column, k):
    # check if the column is numeric
    if df[column].dtype in ["int64", "float64"]:
        # create a KNNImputer object with the specified k
        imputer = KNNImputer(n_neighbors=k)

        # fit the imputer on the data and transform the data to fill the missing values
        df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))
    else:
        # if the column is not numeric, do nothing
        pass
    return df

def replace_missing_values_with_knn(df, k):
    # find all numeric columns in the dataframe
    numeric_columns = df.select_dtypes(["int64", "float64"]).columns

    # iterate over the numeric columns
    for column in numeric_columns:
        # create a KNNImputer object with the specified k
        imputer = KNNImputer(n_neighbors=k)

        # fit the imputer on the data and transform the data to fill the missing values
        df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))

    return df

def replace_missing_values_with_matrix_completion(df, column):
    # check if the column is numeric
    if df[column].dtype in ["int64", "float64"]:
        # create an IterativeImputer object
        imputer = IterativeImputer()

        # fit the imputer on the data and transform the data to fill the missing values
        df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))
    else:
        # if the column is not numeric, do nothing
        pass
    return df

def determine_major_datatype(df, column_name):
    # Count the number of occurrences of each datatype in the given column
    datatype_counts = df[column_name].apply(type).value_counts()
    
    # Return the datatype with the most occurrences
    return datatype_counts.index[0]

def convert_mismatched_datatypes(df, column_name):
    # Determine the major datatype in the given column
    major_datatype = df[column_name].apply(type).value_counts().index[0]
    
    # Return a boolean mask indicating which elements have a datatype that doesn't match the major datatype
    mismatched_datatypes = df[column_name].apply(type) != major_datatype
    
    # Convert the mismatched elements to the major datatype
    df.loc[mismatched_datatypes, column_name] = df[column_name][mismatched_datatypes].astype(major_datatype)
    
    # Return the updated DataFrame
    return df


def verify_dataframe_column(dataframe, column_name, reference_data, key_column):
  # Merge the dataframe and the reference data using a common key
  merged_df = pd.merge(dataframe, reference_data, on=key_column)

  # Define a verification function
  def verify(row):
    # Compare the values in the relevant column with the corresponding
    # values in the reference data, and return True if they match
    if row[column_name] == row["reference_column"]:
      return True
    else:
      return False

  # Apply the verification function to each row in the dataframe
  merged_df["is_verified"] = merged_df.apply(verify, axis=1)

  # Count the number of rows where the verification function returned True
  num_verified = merged_df["is_verified"].sum()

  return num_verified


def remove_trailing_spaces(df):
  # Use the apply method to apply the rstrip function to each value in the dataframe
  clean_df = df.applymap(str.rstrip)

  return clean_df


def standardize_text(df, column_name, standardization_type, max_length=None, pad_string=None, case=None):
  # Create a new dataframe with the same columns as the input dataframe
  standardized_df = pd.DataFrame(columns=df.columns)

  # Iterate over the rows in the input dataframe
  for index, row in df.iterrows():
    # Create a dictionary containing the standardized values for each column
    standardized_row = {}

    # Standardize the value in the specified column using the specified standardization type
    value = row[column_name]

    # Trim the value if the standardization type is "trim"
    if standardization_type == "trim":
      standardized_value = value.strip()

    # Pad the value if the standardization type is "pad"
    elif standardization_type == "pad":
      standardized_value = value.ljust(max_length, pad_string)

    # Convert the case of the value if the standardization type is "case"
    elif standardization_type == "case":
      if case == "upper":
        standardized_value = value.upper()
      elif case == "lower":
        standardized_value = value.lower()
      elif case == "title":
        standardized_value = value.title()

    # Add the standardized value to the dictionary
    standardized_row[column_name] = standardized_value

    # Add the standardized row to the new dataframe
    standardized_df = standardized_df.append(standardized_row, ignore_index=True)

  return standardized_df

def standardize_numeric_values(dataframe, column, rounding=None, scaling_factor=None):
    if rounding:
        dataframe[column] = dataframe[column].round(rounding)
    if scaling_factor:
        dataframe[column] = dataframe[column] * scaling_factor
    return dataframe

def combine_data(left_dataframe, right_dataframe, left_key, right_key, join_type):
    if join_type.lower() == 'inner':
        joined_dataframe = pd.merge(left_dataframe, right_dataframe, left_on=left_key, right_on=right_key, how='inner')
    elif join_type.lower() == 'outer':
        joined_dataframe = pd.merge(left_dataframe, right_dataframe, left_on=left_key, right_on=right_key, how='outer')
    elif join_type.lower() == 'left':
        joined_dataframe = pd.merge(left_dataframe, right_dataframe, left_on=left_key, right_on=right_key, how='left')
    elif join_type.lower() == 'right':
        joined_dataframe = pd.merge(left_dataframe, right_dataframe, left_on=left_key, right_on=right_key, how='right')
    elif join_type.lower() == 'cross':
        joined_dataframe = pd.merge(left_dataframe, right_dataframe, left_on=left_key, right_on=right_key, how='cross')
    else:
        raise ValueError('Invalid join type')
    return joined_dataframe

def combine_data_concat(dataframes, axis):
    return pd.concat(dataframes, axis=axis)


def split_data(dataframe, column, values):
    return dataframe[dataframe[column].isin(values)]


def combine_similar_data(data) -> pd.DataFrame:
    # Concatenate all dataframes into a single one
    df = pd.concat(data)

    # Use MiniBatch K-Means clustering to identify similar data points
    kmeans = MiniBatchKMeans(n_clusters=len(data))
    clusters = kmeans.fit_predict(df)

    # Use TruncatedSVD to reduce the dimensions of the data
    svd = TruncatedSVD(n_components=len(data))
    reduced = svd.fit_transform(df)

    # Create a new dataframe with the reduced dimensions and cluster labels
    new_df = pd.DataFrame(reduced, columns=["dim_{}".format(i) for i in range(len(data))])
    new_df["cluster"] = clusters

    # Group the data by cluster and compute the mean of each cluster
    combined_data = new_df.groupby("cluster").mean()

    return combined_data


def summarize_data(data: pd.DataFrame) -> pd.DataFrame:
    # Select only the numeric columns
    numeric_data = data.select_dtypes(include=["number"])

    # Compute the mean, median, and mode of the numeric data
    mean = numeric_data.mean()
    median = numeric_data.median()
    mode = numeric_data.mode()

    # Compute the standard deviation and interquartile range of the numeric data
    std = numeric_data.std()
    iqr = numeric_data.quantile(0.75) - numeric_data.quantile(0.25)

    # Create a new dataframe with the summary statistics
    summary = pd.concat([mean, median, mode, std, iqr], axis=1)
    summary.columns = ["mean", "median", "mode", "std", "iqr"]

    return summary


def summarize_groups(data: pd.DataFrame, groupby: str, agg: Dict[str, str]) -> pd.DataFrame:
    # Group the data by the specified column
    grouped = data.groupby(groupby)

    # Apply the specified aggregation functions to each group
    summarized = grouped.agg(agg)

    return summarized



def sample_data(data: pd.DataFrame, n: int, stratify: Optional[str] = None, weights: Optional[str] = None) -> pd.DataFrame:
    # Use stratified sampling if a column for stratification is specified
    if stratify:
        groups = data.groupby(stratify)
        sampled = pd.concat([g.sample(min(len(g), n), weights=weights) for _, g in groups])
    # Otherwise, use reservoir sampling
    else:
        sampled = data.sample(n=n, weights=weights)

    return sampled

def transform_data(data: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
    # Apply the specified transformation function to the columns
    if method == "log":
        transformed = data[columns].apply(np.log1p)
    elif method == "z-score":
        transformed = data[columns].apply(lambda x: (x - x.mean()) / x.std())
    elif method == "min-max":
        transformed = data[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        raise ValueError("Invalid transformation method: {}".format(method))

    return transformed

def identify_and_correct_outliers(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Identify the numeric columns
    numeric_columns = data.select_dtypes(include=["number"]).columns

    # Compute the first and third quartiles of the numeric data
    q1 = data[numeric_columns].quantile(0.25)
    q3 = data[numeric_columns].quantile(0.75)

    # Compute the interquartile range (IQR) of the numeric data
    iqr = q3 - q1

    # Identify the outliers using the IQR method
    outliers = data[numeric_columns][(data[numeric_columns] < q1 - 1.5 * iqr) | (data[numeric_columns] > q3 + 1.5 * iqr)]

    # Correct the outliers by capping them at the limits defined by the IQR method
    corrected = data[numeric_columns].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    return corrected, outliers

def convert_dataframe(dataframe, file_format, file_path):
    """Convert a pandas dataframe to any file format.
    
    Args:
        dataframe: The pandas dataframe to be converted.
        file_format: The file format to convert to, such as 'csv', 'excel', 'json', 'html', or 'sql'.
        file_path: The file path where the converted data will be saved.
        
    Returns:
        None
    """
    if file_format == 'csv':
        dataframe.to_csv(file_path)
    elif file_format == 'excel':
        dataframe.to_excel(file_path)
    elif file_format == 'json':
        dataframe.to_json(file_path)
    elif file_format == 'html':
        dataframe.to_html(file_path)
    elif file_format == 'sql':
        dataframe.to_sql(file_path)

def clean_and_preprocess_data(df):
    # split the "name" column into first and last name columns
    df = split_column(df, "name", " ", ["first_name", "last_name"])

    # merge the "address1" and "address2" columns into a single "address" column
    df = merge_columns(df, ["address1", "address2"], ", ", "address")

    # group the rows by the "city" column and calculate the mean of the "price" column for each group
    df = aggregate_by_group(df, "city", {"price": "mean"})

    # add 6 months to the "date" column
    df = add_date_offset(df, "date", 6, "months")

    # extract the year, month, and day from the "date" column and create new columns for each
    df = extract_date_parts(df, "date", ["year", "month", "day"])

    # remove numerical values from the "description" column
    df = remove_numbers(df, "description")

    # remove extra white space from the "description" column
    df = remove_white_spaces(df, "description")

    # remove words that are shorter than 3 characters from the "description" column
    df = remove_short_words(df, "description", 3)

    # lemmatize the words in the "description" column
    df = lemmatize_values(df, "description")

    # stem the words in the "description" column
    df = stem_values(df, "description")

    # return the cleaned and preprocessed DataFrame
    return df

def clean_and_preprocess_data_extra(df):
    # split the "name" column into first and last name columns
    df = split_column(df, "name", " ", ["first_name", "last_name"])

    # merge the "address1" and "address2" columns into a single "address" column
    df = merge_columns(df, ["address1", "address2"], ", ", "address")

    # group the rows by the "city" column and calculate the mean of the "price" column for each group
    df = aggregate_by_group(df, "city", {"price": "mean"})

    # add 6 months to the "date" column
    df = add_date_offset(df, "date", 6, "months")

    # extract the year, month, and day from the "date" column and create new columns for each
    df = extract_date_parts(df, "date", ["year", "month", "day"])

    # remove numerical values from the "description" column
    df = remove_numbers(df, "description")

    # remove extra white space from the "description" column
    df = remove_white_spaces(df, "description")

    # remove words that are shorter than 3 characters from the "description" column
    df = remove_short_words(df, "description", 3)

    # lemmatize the words in the "description" column
    df = lemmatize_values(df, "description")

    # stem the words in the "description" column
    df = stem_values(df, "description")

    # return the cleaned and preprocessed DataFrame
    return df


def clean_numbers(df, columns):
    """Clean the specified numerical columns of the DataFrame by replacing missing values, removing outliers, and normalizing the values."""
    for column in columns:
        # Replace missing values with the mean of the column
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)
        
        # Remove outliers using the Tukey method
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
        
        # Normalize the values using min-max normalization
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val)

    return df


def clean_numeric_columns(df: pd.DataFrame, columns: List[str], decimal_places: Optional[int] = None) -> pd.DataFrame:
    """Clean the specified numerical columns in the DataFrame by handling missing values, removing whitespace, special characters, and outliers, and normalizing and rounding values."""
    # Replace missing values with the mean of the column
    df[columns] = df[columns].fillna(df[columns].mean())
    
    # Remove whitespace and special characters from the columns
    df[columns] = df[columns].apply(lambda x: x.str.strip().str.replace(r'[^\w\s]', ''))
    
    # Remove outliers using the Tukey method
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    
    # Normalize values using min-max normalization
    df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    
    # Round values to the specified number of decimal places
    if decimal_places is not None:
        df[columns] = df[columns].round(decimal_places)
    
    return df
