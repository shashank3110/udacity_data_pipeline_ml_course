import sys
import pandas as pd
import sqlite3
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """ Load data function
    messages_filepath : str :  path to messages data.
    categories_filepath : str : path to categories data.
    """
    # message data
    messages = pd.read_csv(messages_filepath, skiprows=0)
    
    # message categories
    categories = pd.read_csv(categories_filepath, skiprows=0)
    
    # merge messages and categories on id
    df = messages.merge(categories,on='id', how='inner')
    return df


def clean_data(df):
    """Data cleaning function
    df : pd.DataFrame : dataset
    """
    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # extracting categry labels
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0].strip())
    
    categories.columns = category_colnames
    
    # cleaning the column values to get the numeric values
    for column in categories:
        # sets each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1].strip())
        # converting column from string to numeric
        categories[column] = categories[column].astype(float)
          
    # concat category back to df
    df.drop('categories', axis=1, inplace=True) 
    df = pd.concat([df, categories],axis=1) 
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saves data to sqlite data 
    df : str : pd.DataFrame
    database_filename : str : database name e.g. 'data.db'
    """
    table_name = 'disaster_data_cleaned'
    
    # drop the test table in case it already exists
    conn = sqlite3.connect(f'{database_filename}')
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # write the claened data to a SQL table
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index=False)
     


def main():
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