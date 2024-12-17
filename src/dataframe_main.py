import sys
import os

sys.path.append(os.path.abspath("./lib"))

from dataframe_functions import (
    callfunction, 
    read_data, 
    column_filter, 
    query_filter, 
    merge_dataframes, 
    write_to_csv
)

#------------------------------------------------------------
# Þarft fyrst að downloada tveimur file-um af https://figshare.com/articles/dataset/Datasets_for_QAnon_on_Reddit_research_project_/19251581
submission_file = '../data/Hashed_Q_Submissions_Raw_Combined.csv'
user_file =  '../data/Hashed_allAuthorStatus.csv' 

output_file = '../data/PostsOfQAnon.csv' # Ekki breyta
#------------------------------------------------------------

df_submission, df_users = callfunction(read_data, "Reading files", [[submission_file,user_file],-1])
df_submission = callfunction(query_filter, "Filtering text only posts", args=[df_submission,'is_self == True'])
df_submission = callfunction(column_filter, "Filtering submission columns", [df_submission, ['text', 'author']])
df_users = callfunction(column_filter, "Filtering user columns", [df_users, ['isUQ', 'QAuthor']])
df = callfunction(merge_dataframes, "Merging dataframes", [df_submission, df_users, ['author', 'QAuthor'], 'left'])
df = callfunction(query_filter, "Filtering out QAnon interested", args=[df,'isUQ == 1'])
df = callfunction(column_filter, "Filtering final columns",[df,['text','author']])
callfunction(write_to_csv, "Writing to csv", [df, output_file])
