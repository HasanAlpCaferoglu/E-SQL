import os
import json
import string
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from typing import List


def nltk_downloads():
    nltk.download('stopwords') # Download the stopwords
    nltk.download('punkt')  # Download the punkt tokenizer
    nltk.download('punkt_tab') # Download the punkt_tab
    return


def save_dataframe_to_csv(df: pd.DataFrame, path: str):
    """
    Saves the given pandas DataFrame to a CSV file at the specified path.

    Arguments:
    df (pd.DataFrame): The DataFrame to save.
    path (str): The file path where the CSV file will be saved.
    """
    try:
        df.to_csv(path, index=False)  # Set index=False to avoid saving the index column
        print(f"DataFrame saved successfully to {path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the DataFrame: {e}")


def clean_text(textData: str)-> str:
    """
    The function process the given textData by removing stop words, removing punctuation marks and lowercasing the textData.

    Arguments:
        textData (str): text to be cleaned
    
    Returns:
        processedTextData (str): cleaned text
    """

    if isinstance(textData, str):
        textData = textData.lower()
        textData = textData.replace("       ", '')

        # Removing punctuations
        # textData = textData.translate(str.maketrans('', '', string.punctuation)) # converts "don't" to "dont"
        textData = textData.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # converts "don't" to "don t"

        # Removing stopwords
        stopWordsSet = set(stopwords.words('english'))
        tokens = word_tokenize(textData)
        tokens = [token for token in tokens if not token.lower() in stopWordsSet]

        processedTextData = ' '.join(tokens)
        return processedTextData
    else:
    # if the text data is NaN return empty string
        return ''
    

def construct_column_information(table_desc_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    The function combines the original column name, column description, data format, and value description information from table description CSV files into a single descriptive string for each column and adds it as a new column in the DataFrame.

    Arguments:
        table_desc_df (pd.DataFrame): DataFrame containing table descriptions.
        table_name (str): Name of the table.

    Returns:
        pd.Series: constructed single text column information for each column 
    """
    # Function to build column info for each row
    def build_column_info(row):
        column_info = f"The information about the {row['original_column_name']} column of the {table_name} table [{table_name}.{row['original_column_name']}] is as following."

        if pd.notna(row['column_description']):
            column_info += f" The {row['original_column_name']} column can be described as {row['column_description']}."
        if pd.notna(row['value_description']):
            column_info += f" The value description for the {row['original_column_name']} is {row['value_description']}"
        
        column_info = column_info.replace("       ", ' ')
        column_info = column_info.replace("       ", ' ')
        return column_info

    # Apply the function to create the "column_info" column
    # table_desc_df['column_info'] = table_desc_df.apply(build_column_info, axis=1)
    column_info_series = table_desc_df.apply(build_column_info, axis=1)

    return column_info_series


def process_database_descriptions(database_description_path: str):
    """
    Processes multiple CSV files in the given directory, applies the pre-existing construct_column_information function to each,
    and combines the "column_info" columns into a single DataFrame which is then saved as db_description.csv.

    Arguments:
        database_description_path (str): Path to the directory containing database description CSV files.
    """

    # List to store column_info from each file
    all_column_infos = []

    # Iterate over each file in the directory
    for filename in os.listdir(database_description_path):
        if filename.endswith(".csv") and filename != "db_description.csv" :
            print(f"------> {filename} table start to be processed.")
            file_path = os.path.join(database_description_path, filename)
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                
            table_name = filename.replace('.csv', '')
            column_info_series = construct_column_information(df, table_name)
            # Convert the Series to a DataFrame with a single column named 'column_info'
            column_info_df = column_info_series.to_frame(name='column_info')
            all_column_infos.append(column_info_df)

    # Combine all column_info data into a single DataFrame
    all_info_df = pd.concat(all_column_infos, ignore_index=True)

    # Save the DataFrame to a CSV file
    output_path = os.path.join(database_description_path, 'db_description.csv')
    all_info_df.to_csv(output_path, index=False)
    print(f"---> Database information saved successfully to {output_path}")
    
    return



def process_all_dbs(dataset_path: str, mode: str):
    """
    The function processes description of all databases and construct db_description.csv file for all databases.

    Arguments:
        dataset_path (str): General dataset path
        mode (str): Either dev, test or train
    """
    nltk_downloads() # download nltk stop words
    databases_path = dataset_path + f"/{mode}/{mode}_databases"
   
    for db_directory in os.listdir(databases_path):
        print(f"----------> Start to process {db_directory} database.")
        db_description_path = databases_path + "/" + db_directory + "/database_description"
        process_database_descriptions(database_description_path=db_description_path)

    print("\n\n All databases processed and db_description.csv files are created for all.\n\n")
    return


def get_relevant_db_descriptions(database_description_path: str, question: str, relevant_description_number: int = 6) -> List[str]:
    """
    The function returns list relevant column descriptions

    Arguments:
        database_description_path (str): Path to the directory containing database description CSV files.
        question (str): the considered natural language question
        relevant_description_number (int): number of top ranked column descriptions

    Returns:
        relevant_db_descriptions (List[str]): List of relevant column descriptions.
    """
    db_description_csv_path = database_description_path + "/db_description.csv"

    if not os.path.exists(db_description_csv_path):
        process_database_descriptions(database_description_path)
    
    # Read the database description db_info.csv file
    db_desc_df = pd.read_csv(db_description_csv_path)
    db_description_corpus = db_desc_df['column_info'].tolist()
    db_description_corpus_cleaned = [clean_text(description) for description in db_description_corpus]

    # Tokenize corpus and create bm25 instance using cleaned corpus
    tokenized_db_description_corpus_cleaned = [doc.split(" ") for doc in db_description_corpus_cleaned]
    bm25 = BM25Okapi(tokenized_db_description_corpus_cleaned)

    # Tokenize question
    tokenized_question = question.split(" ")

    relevant_db_descriptions = bm25.get_top_n(tokenized_question, db_description_corpus, n=relevant_description_number)
    return relevant_db_descriptions

def get_db_column_meanings(database_column_meaning_path: str, db_id: str) -> List[str]:
    """
    The function extracts required database column meanings.

    Arguments:
        database_column_meaning_path (str): path to the column_meaning.json
        db_id (str): name of the database whose columns' meanings will be extracted

    Returns:
        List[str]: A list of strings explaining the database column meanings.
    """
    # Load the JSON file
    with open(database_column_meaning_path, 'r') as file:
        column_meanings = json.load(file)
    
    # Initialize a list to store the extracted meanings
    meanings = []

    # Iterate through each key in the JSON
    for key, explanation in column_meanings.items():
        # Check if the key starts with the given db_id
        if key.startswith(db_id + "|"):
            # Extract the table name and column name from the key
            _, table_name, column_name = key.split("|")
            # Construct the meaning string in the desired format
            meaning = f"# Meaning of {column_name} column of {table_name} table in database is that {explanation.strip('# ').strip()}"
            meanings.append(meaning)
    
    return meanings
