import pandas as pd
import numpy as np

def callfunction(func, msg, args=None):
    print(msg, end='...', flush=True)
    if args is None:
        out = func()
    else:
        out = func(args)
    print("\033[32mcompleted\033[0m")
    return out

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

def column_filter(args):
    df = args[0]
    columns = args[1]
    return df[columns]

def query_filter(args):
    df = args[0]
    query = args[1]
    return df.query(query) 

def merge_dataframes(args):
    df1 = args[0]
    df2 = args[1]
    columns = args[2]
    method = args[3]
    df = df1.merge(df2, left_on=columns[0], right_on=columns[1], how=method)
    df = df.drop(columns=columns[1])
    df = df.dropna()
    return df

def write_to_csv(args):
    df = args[0]
    path = args[1]
    df.to_csv(path_or_buf = path)
