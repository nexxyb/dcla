from celery import shared_task
#import cleaner
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import redis

def handle_exception(e):
    # Get the logger from the Celery app
    logger = logging.getLogger('celery')
    # Log the error
    logger.error(e)

@shared_task
def remove_duplicates(df):
    
   return df.drop_duplicates()

@shared_task
def remove_columns(df, columns):
    return df.drop(columns, axis=1)

@shared_task
def remove_null_on_column(df, columns):
    # remove null values from the specified column
    for column in columns:
        df = df[pd.notnull(df[column])]
    return df

@shared_task
def replace_null_on_columns(df, value_pairs_list):
    for column_value_pairs in value_pairs_list:  
        for column, value in column_value_pairs:
            df[column] = df[column].fillna(value)
    return df

@shared_task
def convert_data_type(df,data_type_columns_and_types):
    for type_pair in data_type_columns_and_types:
        for data_type_column, data_type in type_pair:
            if data_type == 'float':
                # Convert the column to float
                df[data_type_column] = pd.to_numeric(df[data_type_column], errors='coerce')
            elif data_type == 'integer':
                # Convert the column to integer
                df[data_type_column] = pd.to_numeric(df[data_type_column], errors='coerce').astype(int)
            elif data_type == 'boolean':
                # Convert the column to boolean
                df[data_type_column] = df[data_type_column].map({'True': True, 'False': False})
            elif data_type == 'string':
                # Convert the column to string
                df[data_type_column] = df[data_type_column].astype(str)
            elif data_type == 'datetime':
                # Convert the column to datetime
                df[data_type_column] = pd.to_datetime(df[data_type_column], errors='coerce')
            elif data_type == 'timedelta[ns]':
                # Convert the column to timedelta[ns]
                df[data_type_column] = pd.to_timedelta(df[data_type_column], errors='coerce')
            elif data_type == 'category':
                # Convert the column to category
                df[data_type_column] = df[data_type_column].astype('category')
            elif data_type == 'complex':
                # Convert the column to complex
                df[data_type_column] = df[data_type_column].apply(complex)
    return df

@shared_task
def clean_data(df_json, tasks):
    df = pd.read_json(df_json)
    task_ids = []
    for task in tasks:
        result = task.apply_async(args=( df))
        #request.session[f"{task.split('.')[0]}_id"]=result.id
        # Connect to the Redis server
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.set(f"{task.split('.')[0]}_id", result.id)
        try:
            df = result.get()
        except Exception as e:
            # Handle the exception
            handle_exception(e)
            df = None

    return df
    