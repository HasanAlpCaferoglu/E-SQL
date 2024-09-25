import sqlite3
import random
import logging
import re
import time
import nltk
from nltk.tokenize import word_tokenize
import difflib
from rank_bm25 import BM25Okapi
import sqlglot
from sqlglot import parse, parse_one, expressions
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.expressions import Select
from func_timeout import func_timeout, FunctionTimedOut
from typing import Any, Union, List, Dict, Optional
nltk.download('punkt')

def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.
    
    Arguments:
        db_path (str): The database sqlite file path.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.
        
    Returns:
        resutls: SQL execution results .
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchmany(10)
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
    except Exception as e:
        logging.error(f"Error in execute_sql: {e}\n db_path: {db_path}\n SQL: {sql}")
        raise e
    
def get_db_tables(db_path):
    """
    The function extracts tables in a specific database.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_tables (List[str]): Names of the tables in the database as a list of string
    """
    # Execute query to extract the names of all tables in the database
    try:
    
        tables = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        db_tables = [table_name_tuple[0] for table_name_tuple in tables if table_name_tuple[0] != "sqlite_sequence"]
        return db_tables
    except Exception as e:
        logging.error(f"Error in get_db_tables: {e}")
        raise e
    
def get_db_colums_of_table(db_path: str, table_name: str) -> List[str]:
    """
    The function extractes all column of a table whose name is given
    
    Args:
        db_path (str): The database sqlite file path.
        table_name (str): The name of the table in the database whose columns extracted.
        
    Returns:
        columns_of_table (List[str]): A list of column names.
    """
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        columns_of_table = [row[1] for row in table_info_rows]
        # data_type_of_columns_of_table = [row[2] for row in table_info_rows]
        return columns_of_table
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e

def isTableInDB(db_path: str, table_name: str) -> bool:
    """
    The function checks whether given table name is in the database

    Arguments:
        db_path (str): The database sqlite file path.
        table_name (str): the name of the table that is going to be checked
    
    Returns:
        bool: True if the table in the database, otherwise returns False
    """

    db_tables = get_db_tables(db_path)
    if table_name in db_tables:
        return True
    else:
        return False

def isColumnInTable(db_path: str, table_name: str, column_name: str) -> bool:
    """
    The function checks whether given column name is in the columns of given table

    Arguments:
        db_path (str): The database sqlite file path
        table_name (str): the name of the table
        column_name (str): the name of the column that is going to be checked
    
    Returns:
        bool: True if the given column is among the columns of given table, otherwise returns False
    """

    columns_of_table = get_db_colums_of_table(db_path, table_name)
    if column_name in columns_of_table:
        return True
    else:
        return False

def get_original_schema(db_path: str) -> str:
    """
    The function construct database schema from the database sqlite file.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_schema (str): database schema constructed by CREATE TABLE statements
    """
    # Connecting to the sqlite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to extract the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Dictionary to hold table names and their CREATE TABLE statements
    db_schema_dict = {}

    for table_name_tuple in tables:
        table_name = table_name_tuple[0] # Extracting the table name from the tuple
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_statement = cursor.fetchone()[0]
        db_schema_dict[table_name] = create_statement

    # Close the connection
    cursor.close()
    conn.close()

    db_schema = " "
    for table, create_table_statement in db_schema_dict.items():
        db_schema = db_schema + create_table_statement + "\n"

    return db_schema


def clean_db_schema(db_schema: str) -> str:
    """
    The function cleans the database schema by removing unnecessary whitespaces and new lines,
    ensuring that each column definition or constraint in a table is described on a single line.

    Arguments:
        db_schema (str): original database schema
    Returns:
        cleaned_db_schema (str): cleaned database schema
    """
    # Split the schema into lines
    lines = db_schema.split('\n')
    cleaned_lines = []
    current_statement = []        
    
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()  # Trim any leading/trailing whitespace
        if not line:
            continue  # Skip empty lines
        if "CREATE TABLE" in line:
            line = line + " (" # append '(' to the lines containing CREATE TABLE
            cleaned_lines.append(line) 
            continue 
        if line[0] == '(':
            continue # Skip lines containing '(' 
        if line[0] == ')':
            cleaned_lines.append(line)
            continue 
        if "primary key" in line.lower():
            cleaned_lines[-1] = cleaned_lines[-1] + 'primary key,' # if the current line is PK, add it to the previous line
            continue

        line = line.replace('AUTOINCREMENT', '')
        line = line.replace('DEFAULT 0', '')
        line = line.replace('NOT NULL', '')
        line = line.replace('NULL', '')
        line = line.replace('UNIQUE', '')
        line = line.replace('ON UPDATE', '')
        line = line.replace('ON DELETE', '')
        line = line.replace('CASCADE', '')
        
        line = line.replace('autoincrement', '')
        line = line.replace('default 0', '')
        line = line.replace('not null', '')
        line = line.replace('null', '')
        line = line.replace('unique', '')
        line = line.replace('on update', '')
        line = line.replace('on delete', '')
        line = line.replace('cascade', '')

        # Remove space before commas
        line = re.sub(r'\s*,', ',', line)

        # Ensure one space between column names and their data types
        # Handling multi-word column names enclosed in backticks
        line = re.sub(r'`([^`]+)`\s+(\w+)', r'`\1` \2', line)
        # Handling standard column names
        line = re.sub(r'(\w+)\s+(\w+)', r'\1 \2', line)
        
        cleaned_lines.append(line)
    # Join all cleaned lines into a single string
    cleaned_db_schema = '\n'.join(cleaned_lines)
    return cleaned_db_schema

def get_schema(db_path: str) -> str:
    """
    The function returns cleaned database schema from the database sqlite file.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_schema (str): cleaned database schema constructed by CREATE TABLE statements
    """
    original_db_schema = get_original_schema(db_path)
    db_schema = clean_db_schema(original_db_schema)
    return db_schema

def get_schema_dict(db_path: str) -> Dict:
    """
    The function construct database schema from the database sqlite file in the form of dict.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_schema_dict (Dict[str, Dict[str, str]]): database schema dictionary whose keys are table names and values are dict with column names keys and data type with as values. 
    """
    # Connecting to the sqlite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to extract the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    
    # Dictionary to hold table names and their CREATE TABLE statements
    db_schema_dict = {}

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        table_info = cursor.fetchall() # in table_info, each row indicate (cid, column name, type, notnull, default value, is_PK)
        # print(f"Table {table_name} info: \n", table_info)
        db_schema_dict[table_name] = {col_item[1]: col_item[2] for col_item in table_info}

    # Close the connection
    cursor.close()
    conn.close()
    
    return db_schema_dict

def get_schema_tables_and_columns_dict(db_path: str) -> Dict:
    """
    The function construct database schema from the database sqlite file in the form of dict.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_schema_dict (Dict[str, List[str]]): database schema dictionary whose keys are table names and values are list of column names. 
    """
    # Connecting to the sqlite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to extract the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    
    # Dictionary to hold table names and their CREATE TABLE statements
    db_schema_dict = {}

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        table_info = cursor.fetchall() # in table_info, each row indicate (cid, column name, type, notnull, default value, is_PK)
        # print(f"Table {table_name} info: \n", table_info)
        db_schema_dict[table_name] = [col_item[1] for col_item in table_info]

    # Close the connection
    cursor.close()
    conn.close()
    
    return db_schema_dict

def clean_sql(sql: str) -> str:
    """
    The function removes unwanted whitespace and characters in the given SQL statement
    
    Arguments:
        sql (str): The SQL query.
    Returns:
        clean_sql (str): Clean SQL statement.
    """
    # clean_sql=  sql.replace('\n', ' ').replace('"', "'").strip("`.")
    clean_sql = sql.replace('\n', ' ').replace('"', "`").replace('\"', "`")
    return clean_sql

    

def compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    """
    Compares the results of two SQL queries to check for equivalence.
    
    Args:
        db_path (str): The database sqlite file path.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        
    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.
    
    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        predicted_res = execute_sql(db_path, predicted_sql)
        ground_truth_res = execute_sql(db_path, ground_truth_sql)
        return int(set(predicted_res) == set(ground_truth_res))
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise e
    
    
def compare_sqls(db_path: str, predicted_sql: str, ground_truth_sql: str, meta_time_out: int = 30) -> Dict[str, Union[int, str]]:
    """
    Compares predicted SQL with ground truth SQL within a timeout.
    
    Arguments:
        db_path (str): The database sqlite file path.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        meta_time_out (int): The timeout for the comparison.
        
    Returns:
        dict: A dictionary with the comparison result and any error message.
    """
    predicted_sql = clean_sql(predicted_sql)
    try:
        res = func_timeout(meta_time_out, compare_sqls_outcomes, args=(db_path, predicted_sql, ground_truth_sql))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        logging.warning("Comparison timed out.")
        error = "timeout"
        res = 0
    except Exception as e:
        logging.error(f"Error in compare_sqls: {e}")
        error = str(e)
        res = 0
    return {'exec_res': res, 'exec_err': error}


def extract_sql_tables(db_path: str, sql: str) -> List[str]:
    """
    The function extracts the table names in the SQL.
    
    Args:
        db_path (str): The database sqlite file path.
        sql (str): The ground truth SQL query string.
        
    Returns:
        tables_in_sql (List[str]): Names of the tables in the ground truth SQL query as a list of string.
    """
    db_tables = get_db_tables(db_path)
    try:
        parsed_tables = list(parse_one(sql, read='sqlite').find_all(expressions.Table)) # parsed_tables: List[<sqlglot.expression.Table>] 
        tables_in_sql = [str(table.name) for table in parsed_tables if str(table.name) in [db_table.lower() for db_table in db_tables]] # tables_in_sql: List[str]
        tables_in_sql = list(set(tables_in_sql)) # ensure list contains only unique table values i.e. a table name doesn't repeat in the list
        return tables_in_sql
    except Exception as e:
        logging.critical(f"Error in extract_sql_tables: {e}\n")
        raise e
    
def extract_sql_tables_with_aliases(db_path: str, sql: str) -> List[str]:
    """
    The function extracts the table names with their aliases in the SQL.
    
    Args:
        db_path (str): The database sqlite file path.
        sql (str): The ground truth SQL query string.
        
    Returns:
        tables_w_aliases  (List[Dict[str, str]]): List of dictionary whose keys are "table_name" and "table_alias"
    """
    db_tables = get_db_tables(db_path)
    try:
        parsed_tables = list(parse_one(sql, read='sqlite').find_all(expressions.Table)) # parsed_tables: List[<sqlglot.expression.Table>] 
        tables_w_aliases = [{"table_name": str(table.name), "table_alias": str(table.alias)} for table in parsed_tables if str(table.name) in [db_table.lower() for db_table in db_tables]] # tables_in_sql: List[str]
        tables_w_aliases = [table_alias_dict for table_alias_dict in tables_w_aliases if table_alias_dict['table_alias'] != '']
        return tables_w_aliases
    except Exception as e:
        logging.warning(f"Error in extract_sql_tables_with_aliases: \n\tError{e} \n\t{sql}")
        raise

def replace_alias_with_table_names_in_sql(db_path: str, sql: str) -> str:
    """
    The function removes aliases in the SQL. 

    Arguments:
        sql (str): The SQL with aliases
    Returns:
        sql (str): The SQL without aliases. Table aliases are replaced with corresponding table names.
    """
    try:
        tables_w_aliases = extract_sql_tables_with_aliases(db_path, sql)
        for table_dict in tables_w_aliases:
            table_name, table_alias = table_dict['table_name'], table_dict['table_alias']
            sql = sql.replace(table_alias+".", table_name+".") # replace table_alias with table names in necessary clauses
            # sql = sql.replace(f"AS {table_alias}", "") # remove "AS" keywords
            # sql = sql.replace(table_alias, "") # remove table alias
            sql = re.sub(r'\s+', ' ', sql) # remove extra spaces
        
        return sql
    except Exception as e:
        logging.warning(f"Failed to replace aliases in SQL due to error: {e}. Initially given error is returned.")
        return sql



def extract_sql_columns(db_path: str, sql: str) -> Dict[str, List[str]]:
    """
    The function extracts the column names with corresponding table names in the SQL.
    
    Args:
        db_path (str): The database sqlite file path.
        sql (str): The SQL query as string.
        
    Returns:
        columns_in_sql_dict (Dict[str, List[str]]): A dictionary where keys are table names and values are lists of column names.
    """
    columns_in_sql_dict = {}

    # extract database tables
    db_tables = get_db_tables(db_path)

    # Qualify columns such that columns will always be followed by table_name
    db_schema_dict = get_schema_dict(db_path)
    try:
        qualified_parsed_sql = qualify(parse_one(sql, read='sqlite'), schema=db_schema_dict, qualify_columns=True, validate_qualify_columns=False) 
    except Exception as e:
        logging.critical(f"Error in qualifying parsed SQL in extract_sql_columns function. \n{e} \nSQL: {sql} ")
        qualified_parsed_sql = parse_one(sql, read='sqlite')
    # print("qualified and parsed sql: \n", qualified_parsed_sql)
    """ 
    Type of  qualified_parsed_sql is <class 'sqlglot.expressions.Select'>
    When qualified sql as a string is reauired, use __str__() method of 'sqlglot.expressions.Select' class  
    """

    # Extract tables and its aliases
    tables_w_aliases = extract_sql_tables_with_aliases(db_path, qualified_parsed_sql.__str__())
    # print("tables_w_aliases: \n", tables_w_aliases)
    
    try:
        parsed_columns = list(qualified_parsed_sql.find_all(expressions.Column)) 
        # parsed_columns: List[<sqlglot.expression.Column>]
        # Note that columns and table names will be lower case if it is a single word
        # print("parsed columns: \n", parsed_columns)

        for column_obj in parsed_columns:
            column_name = column_obj.name
            table_name = column_obj.table
            # print("column_name and table_name: ", column_name, "-", table_name )

            for tables_alias_dict in tables_w_aliases:
                if table_name == tables_alias_dict["table_alias"]:
                    table_name = tables_alias_dict["table_name"]
            
            if table_name.lower() in [db_table.lower() for db_table in db_tables]:
                db_columns_of_table = get_db_colums_of_table(db_path, table_name)
                if column_name.lower() in [col.lower() for col in db_columns_of_table]:
                    if table_name in columns_in_sql_dict:
                        if column_name not in columns_in_sql_dict[table_name]:
                            columns_in_sql_dict[table_name].append(column_name)
                    else:
                        columns_in_sql_dict[table_name] = [column_name]

        # reconstruct columns_in_sql_dict so that its items are original database table ans column names
        original_columns_in_sql_dict = {}
        for table_name, col_list in columns_in_sql_dict.items():
            db_tables_lower = [t.lower() for t in db_tables]
            t_index = db_tables_lower.index(table_name)
            db_table_original_name = db_tables[t_index]
            original_columns_in_sql_dict[db_table_original_name] = []
            
            table_cols_original = get_db_colums_of_table(db_path, db_table_original_name)
            table_cols_lower = [c.lower() for c in table_cols_original]
            for col in col_list:
                if col in table_cols_lower:
                    c_index = table_cols_lower.index(col)
                    c_original_name = table_cols_original[c_index]
                    original_columns_in_sql_dict[db_table_original_name].append(c_original_name)
                elif col in table_cols_original:
                    original_columns_in_sql_dict[db_table_original_name].append(col)

        columns_in_sql_dict = original_columns_in_sql_dict                  
        return columns_in_sql_dict
    except Exception as e:
        logging.critical(f"Error in extract_sql_columns:{e}\n")

def generate_schema_from_schema_dict(db_path: str, schema_dict: Dict) -> str:
    """
    The function creates filtered schema string from the given schema dictionary

    Arguments:
        db_path (str): The database sqlite file path.
        schema_dict (Dict[str, List[str]]): A dictionary where keys are table names and values are lists of column names.

    Results:
        schema_str (str): Schema generated from the given schema dictionary and database path
    """
    # Dictionary to store CREATE TABLE statements
    create_statements = []

    for table, column_list in schema_dict.items():
        table_info = execute_sql(db_path, f"PRAGMA table_info(`{table}`);") # returns tuple (cid, name type, notnull, dflt_value, pk) for each column in the taple
        # print(f"TABLE INFO OF {table} \n", table_info)
        pk_columns = [(row[1], row[2]) for row in table_info if row[5] != 0] # add all PKs in schema definition with ther data types
        other_columns = [(row[1], row[2]) for row in table_info if row[5] == 0 and row[1] in column_list] # add column if it is exist in the column list in the given schema dict

        # Query for foreign key information
        foreign_keys_info = execute_sql(db_path, f"PRAGMA foreign_key_list(`{table}`)")
        # print(f"Foreing keys info:\n", foreign_keys_info)
        foreign_keys = {row[3]: (row[2], row[4]) for row in foreign_keys_info if row[3] in column_list}  # local_col: (ref_table, foreing_col) if local_col exist in filtered column list in the given schema dict
        # print("foreign_keys: \n", foreign_keys)

        table_definition = f"CREATE TABLE {table} (\n"
        if len(pk_columns) == 1: 
            pk = pk_columns[0]
            table_definition = table_definition + f"{pk[0]} {pk[1]} primary key, \n"
            for col in other_columns:
                table_definition = table_definition + f"{col[0]} {col[1]},\n"
            for local_col, ref_table_col in foreign_keys.items():
                table_definition = table_definition + f"foreing key ({local_col}) references {ref_table_col[0]}({ref_table_col[1]}) \n"
        elif len(pk_columns) > 1:
            # concatenate primary key column names with their data types
            for pk in pk_columns:
                table_definition = table_definition + f"{pk[0]} {pk[1]}, \n"

            # concatenate columns with their data types
            for col in other_columns:
                table_definition = table_definition + f"{col[0]} {col[1]},\n"

            # concatenate primary key descriptions
            table_definition = table_definition + "primary key ("
            for ind, pk in enumerate(pk_columns):
                if ind < len(pk_columns)-1:
                    table_definition = table_definition + pk[0] + ", "
                else:
                    table_definition = table_definition + pk[0]

            table_definition = table_definition + "),\n"

            # concatenate foreign key descriptions
            for local_col, ref_table_col in foreign_keys.items():
                table_definition = table_definition + f"foreing key ({local_col}) references {ref_table_col[0]}({ref_table_col[1]}) \n"
            

        table_definition = table_definition + ")"
        create_statements.append(table_definition)

    schema_str = '\n'.join(create_statements)
    return schema_str


def extract_db_samples_enriched_bm25(question: str, evidence: str, db_path: str, schema_dict: Dict, sample_limit: int) -> str:
    """
    The function extract distict samples for given schema items from the database by ranking values using BM25.
    Ranking is not done seperately for all values of each table.column

    Arguments: 
        question (str): considered natural language question
        evidence (str): given evidence about the question
        db_path (str): The database sqlite file path.
        schema_dict (Dict[str, List[str]]): Database schema dictionary where keys are table names and values are lists of column names

    Returns:
        db_samples (str): concatenated strings gives samples from each column 
    """
    db_samples = "\n"

    question = question.replace('\"', '').replace("\'", "").replace("`", "")
    question_and_evidence = question + " " + evidence
    tokenized_question_evidence = word_tokenize(question_and_evidence)  

    for table, col_list in schema_dict.items():
        db_samples = db_samples + f"## {table} table samples:\n"
        for col in col_list:
            try:
                col_distinct_values = execute_sql(db_path, f"SELECT DISTINCT `{col}` FROM `{table}`") # extract all distinct values
                col_distinct_values = [str(value_tuple[0]) if value_tuple and value_tuple[0] else 'NULL' for value_tuple in col_distinct_values] 
                if 'NULL' in col_distinct_values:
                    isNullExist = True
                else:
                    isNullExist = False

                # col_distinct_values = [str(value_tuple[0]) for value_tuple in col_distinct_values if value_tuple[0]] # not condiderin NULL values
                # col_distinct_values = [value if len(value) < 400 else value[:300] for value in col_distinct_values]  # if the lenght of value is too long take its 300 character only
                # if average lenght of the column values larger than 600 than use only the first item of the values since large length of the values cause context limit 
                if len(col_distinct_values) > 0:
                    average_length = sum(len(value) for value in col_distinct_values) / len(col_distinct_values)
                else: 
                    average_length = 0
                if average_length > 600:
                    col_distinct_values = [col_distinct_values[0]]
                    
                if len(col_distinct_values) > sample_limit:
                    corpus = col_distinct_values.copy() 
                    corpus = [f'{table} {col} {val}' for val in corpus]
                    tokenized_corpus = [doc.split(" ") for doc in corpus]
                    # tokenized_corpus = [word_tokenize(doc) for doc in corpus]  # takes too much time, so don't use it
                    bm25 = BM25Okapi(tokenized_corpus)
                    
                    col_distinct_values = bm25.get_top_n(tokenized_question_evidence, col_distinct_values, n=sample_limit)
                    if isNullExist:
                        col_distinct_values.append("NULL")
                        
                db_samples = db_samples + f"# Example values for '{table}'.'{col}' column: " + str(col_distinct_values) + "\n"
            except Exception as e:
                sql = f"SELECT DISTINCT `{col}` FROM `{table}`"
                logging.error(f"Error in extract_db_samples_enriched_bm25: {e}\n SQL: {sql}")
                error = str(e)
           
    return db_samples

def construct_tokenized_db_table_value_corpus(db_path: str, schema_dict: Dict):
    """
    Function collects all item for each value in the database as "table_name column_name value", then tokenize it

    Arguments:
        db_path (str): The database sqlite file path.
        schema_dict (Dict[str, List[str]]): Database schema dictionary where keys are table names and values are lists of column names

    Returns:
        tokenized_db_corpus (List[List]): List of tokenized database "table_name column_name value" item
        db_corpus (List[Tuple])
    """
    # generating corpus whose items are tokenized version of "table_name column_name value" for each value and table in the database.
    corpus = []
    db_corpus = []
    for table, col_list in schema_dict.items():
        for col in col_list:
            try:
                col_distinct_values = execute_sql(db_path, f"SELECT DISTINCT `{col}` FROM `{table}`") # extract all distinct values
                col_distinct_values = [str(value_tuple[0]) for value_tuple in col_distinct_values if value_tuple[0]]
                # if average lenght of the column values larger than 600 than use only the first item of the values since large length of the values cause context limit 
                if len(col_distinct_values) > 0:
                    average_length = sum(len(value) for value in col_distinct_values) / len(col_distinct_values)
                else: 
                    average_length = 0
                if average_length > 600:
                    col_distinct_values = [col_distinct_values[0]]
                    
                table_col_value_str = [f"{table} {col} {val}" for val in col_distinct_values]
                corpus.extend(table_col_value_str)
                table_col_value_tuples = [(table, col, val) for val in col_distinct_values]
                db_corpus.extend(table_col_value_tuples)

            except Exception as e:
                logging.error(f"Error in extract_db_samples_enriched_bm25: {e}\n SQL: {sql}")
                sql = f"SELECT DISTINCT `{col}` FROM `{table}`"
                error = str(e)

    # construction bm25 object
    tokenized_db_corpus = [doc.split(" ") for doc in corpus]
    # tokenized_db_corpus = [word_tokenize(doc) for doc in corpus if doc]  # takes too much time, so don't use it
    return tokenized_db_corpus, db_corpus

def find_most_similar_table(problematic_t_name: str, db_tables: List[str]) -> str:
    """
    Helper function to find the most similar table name in the database.
    As a string similarity metric, Levenshtein distance  is used
    This helper function calculates the similarity ratio between two strings based on the number of single-character edits (insertions, deletions, substitutions) needed to transform one string into the other.
    This ratio is a value between 0 and 1, where 1 means the strings are identical, and 0 means they are completely different.

    Arguments:
        problematic_t_name (str): name of the table that is not in the database
        db_tables (List[str]): list of database tables

    Returns:
        most_similar_table (str): the name of the table that is actually in the database and most similar to the given problematic table name
    """
    
    similarity_scores = [(t_name, difflib.SequenceMatcher(None, problematic_t_name, t_name).ratio()) for t_name in db_tables]
    most_similar_table = max(similarity_scores, key=lambda x: x[1])[0]
    return most_similar_table

def filtered_schema_correction(db_path: str, filtered_schema_dict: Dict) -> Dict:
    """
    The function checks whether something mismatch with the original schema or not. If there is mismatch, it corrects it.

    Arguments: 
        db_path (str): The database sqlite file path
        filtered_schema_dict (Dict[str, List[str]]): A dictionary where keys are table names and values are list of column names

    Returns:
        final_filtered_schema_dict (Dict[str, List[str]]): Finalized filtered schema dictionary 
        filtered_schema_problems (str): A string that expresses all mismatches
    """
    filtered_schema_problems = ""

    db_tables = get_db_tables(db_path)
    ## Step 1: Check if the tables in filtered schema dictionary are in the database and replace them with the most similar table names

    problematic_tables = []
    for t_name in filtered_schema_dict.keys():
        isInDb = isTableInDB(db_path=db_path, table_name=t_name)
        if not isInDb:
            problematic_tables.append(t_name)
    
    
    if problematic_tables:
        print(f"There is mismatch between database and filtered schema tables. The problematic tables are: {problematic_tables}")
        filtered_schema_problems = filtered_schema_problems + f"There is mismatch between database and filtered schema tables. The problematic tables are: {problematic_tables}"
        for problematic_t_name in problematic_tables:
            most_similar_table = find_most_similar_table(problematic_t_name, db_tables)
            filtered_schema_dict[most_similar_table] = filtered_schema_dict.pop(problematic_t_name)

    ## Step 2: Check if the columns of a table in filtered schema dictiionary are actually column of the table. If not, find new table for them.
    
    for table_name in filtered_schema_dict.keys():
        problematic_columns = {} # Dict[str, bool] --> keys are column names and values are boolean that indicates whether a table containing that column is found
        for column_name in filtered_schema_dict[table_name]:
            isInTable = isColumnInTable(db_path=db_path, table_name=table_name, column_name=column_name)
            if not isInTable:
                print(f"There is a mismatch in filtered schema table columns. {column_name} is not actually in the {table_name} table.")
                filtered_schema_problems = filtered_schema_problems + f"There is a mismatch in filtered schema table columns. {column_name} is not actually in the {table_name} table."
                problematic_columns[column_name] = False # boolean variable indicates whether a table containing that column is found

        # finding tables for problematic columns
        table_column_dict = {}
        db_tables = get_db_tables(db_path)
        for p_column in problematic_columns.keys():
            for db_table in db_tables:
                columns_of_table = get_db_colums_of_table(db_path=db_path, table_name=db_table)
                if p_column in columns_of_table:
                    problematic_columns[p_column] = True
                    if db_table in table_column_dict:
                        table_column_dict[db_table].append(p_column)
                    else:
                        table_column_dict[db_table] = [p_column] 

        
        # constructing final filtered schema
        final_filtered_schema_dict = filtered_schema_dict.copy()

        # removing the problematic columns whose actual tables are found
        for p_column, actual_table_found in problematic_columns.items():
            if actual_table_found:
                final_filtered_schema_dict[table_name].remove(p_column)

        # Appending problematic columns actual table into the final filtered schema dict 
        for actual_table, actual_columns in table_column_dict.items():
            if actual_table in final_filtered_schema_dict:
                final_filtered_schema_dict[actual_table] = final_filtered_schema_dict[actual_table] + actual_columns
            else:
                final_filtered_schema_dict[actual_table] = actual_columns
    

    return final_filtered_schema_dict, filtered_schema_problems


def find_similar_values_incolumn_via_like(db_path: str, table: str, column: str, value: str) -> List[str]:
    """
    This function finds similar values to the given value using SQL LIKE clause in given table and column.

    Arguments:
        db_path (str): The database sqlite file path.
        table (str): Table in the given database.
        column (str): Column belongs to the given table.
        value (str): Given value on which similar values are extracted 

    Returns:
        similar_values (List[str]): List of string which are similar to given value.
    """
    # if the length of value is 1, then just return itself to prevent lots of unnecessary match
    if len(value) == 1:
        return []
    # Observing a single value from column. If its length is larger than 300, then return empty list
    value_observation_sql = f"SELECT `{column}` FROM `{table}` LIMIT 3"
    observed_values = execute_sql(db_path=db_path, sql=value_observation_sql)
    
    if not observed_values:
        return []
  
    observed_values = [str(row[0]) for row in observed_values]
    observed_values_avg_len = sum( map(len, observed_values) ) / len(observed_values)
    if observed_values_avg_len >= 300:
        return []
    
    sql = 'SELECT DISTINCT `{C}` FROM `{T}` WHERE `{C}` LIKE "%{V}%"'.format(C=column, T=table, V=value)
    try:
        similar_values = execute_sql(db_path=db_path, sql=sql)
        similar_values = [str(row[0]) for row in similar_values if len(str(row[0])) < 50]
        # Decrease the size of similar values if there are too much
        if len(similar_values) > 5:
            similar_values = similar_values[:5]
    except Exception as e:
        similar_values = []
        logging.critical(f"Error in finding similar values for a value: {e} \n SQL: {sql}")

    return similar_values
    

def find_similar_values_indb_via_like(db_path: str, value:str) -> Dict:
    """
    This function finds similar values to the given value using SQL LIKE clause in all database.

    Arguments:
        db_path (str): The database sqlite file path.
        value (str): Given value on which similar values are extracted 

    Returns:
        similar_values_dict (Dict[Dict[str, List[str]]]): Tables are the keys of dict, and values of table keys are another dictionary whose keys are column names and values are list of real column values similar to the given value 
    """
    db_tables_columns_dict = get_schema_tables_and_columns_dict(db_path=db_path)
    similar_values_dict = {}
    for table, columns_list in db_tables_columns_dict.items():
        for column in columns_list:
            similar_values_list = find_similar_values_incolumn_via_like(db_path, table, column, value)
            if similar_values_list:
                if table in similar_values_dict:
                    similar_values_dict[table][column] = similar_values_list
                else:
                    similar_values_dict[table] = {}
                    similar_values_dict[table][column] = similar_values_list
    
    return similar_values_dict

def extract_comparison_conditions_in_where_clause(db_path: str, where_clause) -> List[Dict[str, str]]:
    """"
    The function extracts list of dict which describe a condition in WHERE clause

    Arguments
        where_clause (sqlglot.expressions.Where)

    Returns
        conditions_list (List[Dict[str, str]]): List of Dict whose keys are table, column, operation, value

    """

    conditions_list = []
    if not where_clause:
        return conditions_list
    
    # Check if where_clause.this is a composite condition (AND/OR)
    if isinstance(where_clause.this, sqlglot.expressions.And) or isinstance(where_clause.this, sqlglot.expressions.Or):
        where_conditions = list(where_clause.this.flatten()) # flattening the where clause in the case of it is composite condition
    else:
        where_conditions = [where_clause.this]
    
    for cond_ind, condition in enumerate(where_conditions):
        # print("--cond_ind: ", cond_ind)
        # print("--condition: ", condition)
        # print("--condition type: ", type(condition))
        columns_in_where_clause = list(condition.find_all(sqlglot.expressions.Column))
        # print("******columns_in_where_clause: ", columns_in_where_clause)
        # it there is no sqlglot.expressions.Column in the AST of Where clause
        if not columns_in_where_clause:
            # check if the condition.this type is Dot not Column
            if isinstance(condition.this, sqlglot.expressions.Dot):
                try: 
                    if isinstance(condition.this.left, sqlglot.expressions.Literal):
                        dot_table = condition.this.left.this
                    if isinstance(condition.this.right, sqlglot.expressions.Literal):
                        dot_column = condition.this.right.this
                    if isinstance(condition.expression, sqlglot.expressions.Literal):
                        dot_value = condition.expression.this

                    if isinstance(condition, sqlglot.expressions.EQ):
                        op = " = "
                    if isinstance(condition, sqlglot.expressions.NEQ):
                        op = " != "
                    if isinstance(condition, sqlglot.expressions.GT):
                        op = " > "
                    if isinstance(condition, sqlglot.expressions.GTE):
                        op = " >= "
                    if isinstance(condition, sqlglot.expressions.LT):
                        op = " < "
                    if isinstance(condition, sqlglot.expressions.LTE):
                        op = " <= "
                    conditions_list.append({
                        "table": dot_table,
                        "column": dot_column,
                        "op": op,
                        "value": dot_value,
                    })
                except Exception as e:
                    logging.warning("Column in condition couldn't be found. Expression.Dot is found under condition but it couldn't be seperated to its table, column and value")

        for cols in columns_in_where_clause:
            comparison_conditions = [] 
            op = "" 
            EQ_condition = cols.find_ancestor(sqlglot.expressions.EQ)
            if EQ_condition and isinstance(EQ_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(EQ_condition)
                op = " = "
            NEQ_condition = cols.find_ancestor(sqlglot.expressions.NEQ)
            if NEQ_condition and isinstance(NEQ_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(NEQ_condition)
                op = " != "
            GT_condition = cols.find_ancestor(sqlglot.expressions.GT)
            if GT_condition and isinstance(GT_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(GT_condition)
                op = " > "
            GTE_condition = cols.find_ancestor(sqlglot.expressions.GTE)
            if GTE_condition and isinstance(GTE_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(GTE_condition)
                op = " >= "
            LT_condition = cols.find_ancestor(sqlglot.expressions.LT)
            if LT_condition and isinstance(LT_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(LT_condition)
                op = " < "
            LTE_condition = cols.find_ancestor(sqlglot.expressions.LTE)
            if LTE_condition and isinstance(LTE_condition.left, sqlglot.expressions.Column):
                comparison_conditions.append(LTE_condition)
                op = " <= "
            
            for cond in comparison_conditions:
                if not isinstance(cond.left.table, str):
                    continue
                if not isinstance(cond.left.this.this, str):
                    continue
                if not isinstance(cond.right.this, str):
                    continue
                conditions_list.append({
                    "table": cond.left.table,
                    "column": cond.left.this.this,
                    "op": op,
                    "value": cond.right.this,
                })

    # print("conditions_list: \n", conditions_list)  
    return conditions_list

def get_comparison_conditions_from_sql(db_path: str, sql: str) -> List[Dict[str, str]]:
    """"
    The functions extracts conditions in given SQL

    Argumnets:
        db_path (str): The database sqlite path 
        sql (str): Given structured query language

    Returns:
        conditions_dict_list (List[Dict[str, str]]): List of comparison conditions described as dictionary.  Dictionary keys are "table", "column", "op" and "value"
    """
    db_schema_dict = get_schema_dict(db_path)
    
    
    # Qualify columns such that columns will always be followed by table_name
    # Attempting to qualify the SQL
    try:
        qualified_parsed_sql = qualify(parse_one(sql, read='sqlite'), schema=db_schema_dict, qualify_columns=True, validate_qualify_columns=False)
    except Exception as e1:
        logging.warning(f"First attempt to qualify SQL failed. Trying with first replacement set. \n\tError: {e1} \n\tSQL: {sql}")
        # First replacement attempt
        try:
            changed_sql_1 = sql.replace('`', '"')
            changed_sql_1 = changed_sql_1.replace("'", '"')
            qualified_parsed_sql = qualify(parse_one(changed_sql_1, read='sqlite'), schema=db_schema_dict, qualify_columns=True, validate_qualify_columns=False)
        except Exception as e2:
            logging.warning(f"Second attempt to qualify SQL failed. Trying with second replacement set. \n\tError: {e2} \n\tSQL: {changed_sql_1}")
            # Second replacement attempt
            try:
                changed_sql_2 = sql.replace('`', "'")
                changed_sql_2 = changed_sql_2.replace('"', "'")
                qualified_parsed_sql = qualify(parse_one(changed_sql_2, read='sqlite'), schema=db_schema_dict, qualify_columns=True, validate_qualify_columns=False)
            except Exception as e3:
                logging.warning(f"Third attempts to qualify SQL failed.Trying with only parsing the SQL. \n\tError: {e3} \n\tSQL: {changed_sql_2}")
                # Third replacement attempt
                try:
                    changed_sql_3 = sql.replace('`', "'")
                    changed_sql_3 = changed_sql_3.replace('"', "'")
                    qualified_parsed_sql = parse_one(changed_sql_3, read='sqlite')
                except Exception as e4:
                    logging.warning(f"Fourth attempts to qualify SQL failed. Triying with only parsing the SQL. \n\tError: {e4} \n\tSQL: {changed_sql_3}")
                    try:
                        changed_sql_4 = sql.replace('`', '"')
                        changed_sql_4 = changed_sql_4.replace("'", '"')
                        qualified_parsed_sql = parse_one(changed_sql_4, read='sqlite') 
                    except Exception as e5:
                        logging.warning(f"Fifth attempts to qualify SQL failed. Triying with only parsing the SQL. \n\tError: {e4} \n\tSQL: {changed_sql_4}")
                        conditions_dict_list = []
                        return conditions_dict_list
    # Replacing table aliases with actual table names
    # qualified_parsed_sql = replace_alias_with_table_names_in_sql(db_path, qualified_parsed_sql) 

    # print("-qualified_parsed_sql: ", repr(qualified_parsed_sql))
    # Extract the WHERE clauses
    where_clauses = list(qualified_parsed_sql.find_all(sqlglot.expressions.Where)) # where_clauses: List[<class 'sqlglot.expressions.Where'>]
    conditions_dict_list = []
    # print("where_clauses: \n", where_clauses)
    if where_clauses:
        # Extract and print each WHERE clause
        for index, where_clause in enumerate(where_clauses):
            try:
                conditions_list = extract_comparison_conditions_in_where_clause(db_path, where_clause) # List[Dict[str, str]]
                conditions_dict_list.extend(conditions_list)
            except Exception as e:
                logging.critical(f"Error in extracting equality conditions in where clause. \nError: {e} \nSQL: {sql} ")
        
    return conditions_dict_list

def extend_conditions_dict_list(conditions_dict_list: List[Dict[str, str]])-> List[Dict[str,str]]:
    """
    The functions splits all the values in the conditions, then it extends the list by adding each words of a value to the conditions_dict_list
    
    Arguments:
        conditions_dict_list (List[Dict[str,str]]): List of comparison conditions described as dictionary. Dictionary keys are "table", "column", "op" and "value".

    Returns 
        extended_conditions_dict_list  (List[Dict[str,str]]): List of comparison conditions described as dictionary. Dictionary keys are "table", "column", "op" and "value".

    """
    extended_conditions_dict_list = conditions_dict_list.copy()
    for cond_dict in conditions_dict_list:
        table = cond_dict['table']
        column = cond_dict['column']
        op = cond_dict['op']
        value = cond_dict['value']
        splitted_value = value.split()
        if len(splitted_value) > 1:
            for val in splitted_value:
                new_condition_dict = {
                    "table": table,
                    "column": column,
                    "op": op,
                    "value": val
                }
                extended_conditions_dict_list.append(new_condition_dict)

    return extended_conditions_dict_list


def get_extended_comparison_conditions_from_sql(db_path: str, sql: str) -> List[Dict[str, str]]:
    """"
    The functions extracts conditions in given SQL

    Argumnets:
        db_path (str): The database sqlite path 
        sql (str): Given structured query language

    Returns:
        extended_conditions_dict_list (List[Dict[str, str]]): List of comparison conditions described as dictionary.  Dictionary keys are "table", "column", "op" and "value"
    """
    conditions_dict_list = get_comparison_conditions_from_sql(db_path=db_path, sql=sql)
    extended_conditions_dict_list = extend_conditions_dict_list(conditions_dict_list)
    return extended_conditions_dict_list


     
def collect_possible_conditions(db_path: str, sql: str) -> List[Dict[str, Union[str, Dict]]]:
    """
    The functions collects possible where clause conditions depending on the Where clause comparison conditions.

    Arguments:
        db_path (str): The database sqlite path 
        sql (str): Given structured query language

    Returns:
        conditions_dict_list (List[Dict[str, Union[str, Dict]]]): List of comparison conditions described as dictionary
    """
    possible_conditions_dict_list = []
    # comp_conditions_dict_list = get_comparison_conditions_from_sql(db_path, sql) # old versioin
    comp_conditions_dict_list = get_extended_comparison_conditions_from_sql(db_path, sql)
    for comp_cond in comp_conditions_dict_list:
        value = comp_cond['value']
        similar_values_dict = find_similar_values_indb_via_like(db_path, value)
        comp_cond['similar_values'] = similar_values_dict
        possible_conditions_dict_list.append(comp_cond)

    return possible_conditions_dict_list


def measure_execution_time(db_path, query):
    start_time = time.time()
    query_result = execute_sql(db_path, query)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, query_result