"""
Process Data Script
This scripti is part of  Udacity Data Scientist Nanodegree Program  2: Disaster Response Pipeline
To run the script, type or copy/pase line below:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse_Processed.db
The script takes following arguments:
    1) Input  File 1: disaster_messages.csv         - CSV file containing messages - provided by FigureEight
    2) Input  File 2: disaster_categories.csv       - CSV file containing categories - provided by FigureEight
    3) Output File:   DisasterResponse_Processed.db - SQLite database - created by the script
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data function does following:
    1. reads input csv file containing messages
    2. reads input csv file containing categories
    3. Merges the messages and categories datasets using the common id
    4. Assigns the combined dataset to a new dataframe 'df'
    
    Arguments:
        messages_filepath   - path to messages   csv file
        categories_filepath - path to categories csv file
    Output:
        df -> Returns merged data as Pandas DataFrame
    """
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df 

def clean_data(df):
    """
    Clean Data function does following:
    1. Splits categories into separate category columns
    2. Converts category values to just numbers 0 or 1
    Arguments:
        df - raw data Pandas DataFrame
    Outputs:
        df - cleaned data Pandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)

    # select the first row of the categories dataframe
    # extract a list of new column names for categories.
    # rename the columns of `categories`
    firstrow = categories.loc[0,:].values
    category_colnames  = [x[:-2] for x in firstrow]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop rows with value other than 0 or 1 for column 'related' - that is not binary
    df.drop(df[df['related'] == 2].index, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save Data function foes following:
    1. Saves the clean dataset into a sqlite database
    
    Arguments:
        df                - Cleaned data Pandas DataFrame
        database_filename - SQLite database file (.db) destination path
    """
   
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message_categories', engine, index=False, if_exists = 'replace')
    pass 


def main():
    """
    Data Processing Main function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """
    print(sys.argv)
    
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()