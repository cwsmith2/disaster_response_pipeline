import sys
import pandas as pd
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_pipeline.log"),
        logging.StreamHandler()
    ]
)

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets.
    
    Args:
        messages_filepath: string. Filepath for csv file containing messages dataset.
        categories_filepath: string. Filepath for csv file containing categories dataset.
        
    Returns:
        df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    logging.info(f"Loading data from {messages_filepath} and {categories_filepath}")
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    logging.info(f"Loaded {len(df)} records after merging datasets.")
    return df

def clean_data(df):
    """
    Clean the merged dataframe.
    
    Args:
        df: dataframe. Dataframe containing merged content of messages and categories datasets.
        
    Returns:
        df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    logging.info("Cleaning data...")
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
    original_len = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {original_len - len(df)} duplicate records.")
    return df

def save_data(df, database_filepath):
    """
    Save cleaned data into an SQLite database.
    
    Args:
        df: dataframe. Dataframe containing cleaned version of messages and categories data.
        database_filepath: string. Filepath for output database.
    """
    logging.info(f"Saving data to database at {database_filepath}")
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')
    logging.info("Data successfully saved.")

def main():
    try:
        if len(sys.argv) == 4:
            messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

            logging.info("Starting ETL pipeline...")
            df = load_data(messages_filepath, categories_filepath)

            df = clean_data(df)

            save_data(df, database_filepath)
            logging.info("ETL pipeline completed successfully!")
        else:
            logging.error("Incorrect arguments provided.")
            print('Usage: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logging.error(f"Empty data encountered: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()