import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets.
    
    Args:
        messages_filepath: string. Filepath for csv file containing messages dataset.
        categories_filepath: string. Filepath for csv file containing categories dataset.
        
    Returns:
        df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataframe.
    
    Args:
        df: dataframe. Dataframe containing merged content of messages and categories datasets.
        
    Returns:
        df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop original categories column and concatenate new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filepath):
    """
    Save cleaned data into an SQLite database.
    
    Args:
        df: dataframe. Dataframe containing cleaned version of messages and categories data.
        database_filepath: string. Filepath for output database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()