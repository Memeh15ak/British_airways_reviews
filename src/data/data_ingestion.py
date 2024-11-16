import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import requests
from bs4 import BeautifulSoup
import os
import yaml
import logging

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)sss - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        logger.info(f"Successfully read {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while reading {path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the file: {e}")

def store(path : str, file_n : str, df: pd.DataFrame):
    try:
        os.makedirs(path, exist_ok=True)
        file_name = file_n
        full_path = os.path.join(path, file_name)
        df.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")
    except Exception as e:
        logger.error(f"Error: An error occurred while saving data to {path}: {e}")
        raise
logger.debug('data stored sucessfully')

def main():
    try:
       path='raw/raw_data.csv'
       df=load(path)
       
       path='data/raw'
       file_n='raw.csv'
       store(path,file_n,df)
    
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()
