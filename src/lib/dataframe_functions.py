import pandas as pd
import numpy as np

"""
A function used to call other function (this is poorly implemented command pattern).

Args: func, str, list - function to call, message to display when function is called, list of arguements the function takes in.
Returns: output of function called
"""
def callfunction(func, msg, args=None):
    print(msg, end='...', flush=True)
    if args is None:
        out = func()
    else:
        out = func(args)
    print("\033[32mcompleted\033[0m")
    return out

"""
reads two .csv files and creates a pandas dataframe from them.

Args: list, int - the list contains the paths [submission_file_path, user_file_path], n is how many rows to read (if -1 then read entire file)
Returns: dataframes
"""
def read_data(args):
    submission_file, user_file = args[0]
    n = args[1]
    if n > 0:
        df_submission = pd.read_csv(submission_file,nrows=n)
        df_users = pd.read_csv(user_file,nrows=n)
    else:
        df_submission = pd.read_csv(submission_file)
        df_users = pd.read_csv(user_file)
    return df_submission, df_users

"""
Filters out columns of a dataframe

Args: dataframe, columns - df, columns
Returns: dataframe created from the columns of df
"""
def column_filter(args):
    df = args[0]
    columns = args[1]
    return df[columns]

"""
Executes a query on a dataframe

Args: dataframe, query  
Returns: filtered dataframe
"""
def query_filter(args):
    df = args[0]
    query = args[1]
    return df.query(query) 


"""
joins two dataframes

Args: dataframe1, dataframe2, columns to join on, method of joining (left, right, etc)
Returns: joined dataframe
"""
def merge_dataframes(args):
    df1 = args[0]
    df2 = args[1]
    columns = args[2]
    method = args[3]
    df = df1.merge(df2, left_on=columns[0], right_on=columns[1], how=method)
    df = df.drop(columns=columns[1])
    df = df.dropna()
    return df


"""
writes a dataframe to as csv file on a given path

Args: dataframe, path 
"""
def write_to_csv(args):
    df = args[0]
    path = args[1]
    df.to_csv(path_or_buf = path)
