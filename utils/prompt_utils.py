import os
import random
import json
from .db_utils import *
from .retrieval_utils import *
from typing import Any, Union, List, Dict


def load_few_shot_data(few_shot_data_path="../few-shot-data/question_enrichment_few_shot_examples.json"):
    """
    The function returns question enrichment few-shot data completely as a python dict

    Arguments:
        -
    Returns:
        question_enrichment_few_shot_data_dict (dictionary): question enrichment few-shot data completely
    """
    with open(few_shot_data_path, 'r') as file:
        question_enrichment_few_shot_data_dict = json.load(file)

    return question_enrichment_few_shot_data_dict

def sql_possible_conditions_prep(possible_conditions_dict_list: Dict)-> str:
    """
    The function construct conditions statements and concatenate them.

    Arguments:
        possible_conditions_dict_list (List[Dict[str, Union[str, Dict]]]):

    Returns:
        all_possible_conditions (str)
    """
    all_possible_conditions_list = []
    if not possible_conditions_dict_list:
        return ""
    for p_cond in possible_conditions_dict_list:
        condition = f"`{p_cond['table']}`.`{p_cond['column']}` {p_cond['op']} `{p_cond['value']}`"
        all_possible_conditions_list.append(condition)
        similars_dict = p_cond['similar_values']
        if similars_dict:
            for table_name, col_val_dict in similars_dict.items():
                for column_name, value_list in col_val_dict.items():
                    for val in value_list:
                        new_possible_cond = f"`{table_name}`.`{column_name}` {p_cond['op']} `{val}`"
                        all_possible_conditions_list.append(new_possible_cond)

    return str(all_possible_conditions_list)


def question_relevant_descriptions_prep(database_description_path, question, relevant_description_number)-> str:
    """
    The functions concatenate the relevant database item (column) descriptions and returns it as a string

    Arguments:
        database_description_path (str): Path to the directory containing database description CSV files.
        question (str): the considered natural language question
        relevant_description_number (int): number of top ranked column descriptions

    Returns:
        str: Concatenated relevant database item describtions

    """

    relevant_db_descriptions = get_relevant_db_descriptions(database_description_path, question, relevant_description_number)
    db_descriptions_str = ""

    for description in relevant_db_descriptions:
        db_descriptions_str = db_descriptions_str + f"# {description} \n"

    return db_descriptions_str


def db_column_meaning_prep(database_column_meaning_path: str, db_id: str)-> str:
    """
    The functions concatenate the database item (column) descriptions and returns it as a string

    Arguments:
        database_column_meaning_path (str): path to the column_meaning.json
        db_id (str): name of the database whose columns' meanings will be extracted

    Returns:
        str: Concatenated column meanings of given database

    """

    db_column_meanings = get_db_column_meanings(database_column_meaning_path, db_id)
    db_column_meanings_str = ""

    for col_meaning in db_column_meanings:
        db_column_meanings_str = db_column_meanings_str + f"{col_meaning} \n"

    return db_column_meanings_str


def sql_generation_and_refinement_few_shot_prep(few_shot_data_path: str, q_db_id: str, level_shot_number:str, schema_existance: bool, mode: str) -> str:
    """
    The function selects the given number of exemple from the few-shot data, and then concatenate the selected question and their ground-truth SQL in string format. 
    - The few-shot examples will be selected from the set of data contains databases different than the considered question.
    - If the question is already annotated, then few-shot examples will be selected from the set of data in which given question is excluded.
    - Level shot number can be between 0 to 10.

    Arguments:
        ew_shot_data_path (str): few-shot data path
        q_db_id (str): database ID (database name) of considered question
        level_shot_number (int): number of exemple desired to add in the prompt for each level. 
        schema_existance (bool): Whether the schema will be provided for the exemplars in the prompt. If it is True, then schema will be provided.
        mode (str): dev mode or test mode
    Returns:
        few_shot_exemplars (str): selected and concatenated exemplars for the prompt
    """
    bird_sql_path = os.getenv('BIRD_DB_PATH')

    if level_shot_number == 0:
        return ""
    
    few_shot_exemplars = ""
    # Check level_shot_number
    if level_shot_number < 0 or level_shot_number > 10:
        raise ValueError("Invalid few-shot number. The level_shot_number should be between 0 and 10")
    
    # Check schema_existance
    if not isinstance(schema_existance, bool): 
        raise TypeError("Provided variable is not a boolean.")
    
    # Check mode
    if mode not in ['test', 'dev']:
        raise ValueError("Invalid value for mode. The variable must be either 'dev' or 'test'.")

    # Get all few-shot exemples
    all_few_shot_data = load_few_shot_data(few_shot_data_path=few_shot_data_path)
    # Set difficulty levels
    levels = ['simple', 'moderate', 'challanging']

    for level in levels:
        examples_in_level = all_few_shot_data[level]
        selected_indexes = []
        if mode == "dev":
            # remove the annotated questions if their db_id is the same with the considered question's db_id
            examples_in_level_tmp = []
            for example in examples_in_level:
                if q_db_id != example['db_id']:
                    examples_in_level_tmp.append(example)
            examples_in_level = examples_in_level_tmp 
            # By removing the db_ids that is same with the considered question's db_id, the selection of the same question as an example is prevented in the case of the question was in the annotated data
        
        selected_indexes = random.sample(range(0,len(examples_in_level)), level_shot_number) # randomly select exemple_num_for_each_level number of example
        for ind in selected_indexes:
            current_question_info_dict = examples_in_level[ind]
            curr_q_db_id = current_question_info_dict['db_id']
            db_path = bird_sql_path + f"/{mode}/{mode}_databases/{curr_q_db_id}/{curr_q_db_id}.sqlite"
            curr_sql = current_question_info_dict['SQL']

            if schema_existance:
                sql_schema_dict = extract_sql_columns(db_path, curr_sql)
                schema = generate_schema_from_schema_dict(db_path, sql_schema_dict) # filtered schema should be as an example because given schema will be filtered one
                few_shot_exemplars = few_shot_exemplars + "Database Schema: \n" +  schema + '\n'
            
            few_shot_exemplars = few_shot_exemplars + "Question: " + current_question_info_dict['question'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Evidence: " + current_question_info_dict['evidence'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "SQL: " + current_question_info_dict['SQL'] + "\n\n"

    return few_shot_exemplars


def fill_candidate_sql_prompt_template(template: str, schema: str, db_samples: str, question: str, few_shot_examples: str = "", evidence: str = "", db_descriptions: str = "") -> str:
    """
    The functions completes the prompt template by filling the necessary slots which are few_shot_examples, schema, question and evidence 

    Arguments:
        template (str): The template that is going to be filled
        schema (str): The schema of the database to which considered question belong
        questoin (str): The considered question that is going to be enriched
        few_shot_examples (str): few-shot examples that are injected to the prompt
        evidence (str): Given evidence statment if exist
        db_descriptions (str): Question relevant database item(column) descriptions

    Returns:
        prompt (str): Completed prompt for question enrichment
    """
    if evidence == '' or evidence == None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"

    if few_shot_examples == '' or few_shot_examples == None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"

    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    db_samples = "\n### Database Samples: \n\n" + db_samples
    question = "\n### Question: \n" + question

    prompt = template.format(
        FEWSHOT_EXAMPLES = few_shot_examples,
        SCHEMA = schema,
        DB_SAMPLES = db_samples,
        QUESTION = question,
        EVIDENCE = evidence,
        DB_DESCRIPTIONS = db_descriptions
    )

    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt

def extract_question_enrichment_prompt_template(enrichment_template_path = "../prompt_templates/question_enrichment_prompt_template.txt") -> str:
    """
    The function returns the question enrichment prompt template by reading corresponding txt file
    
    Arguments:
        None
    Returns:
        enrichment_prompt_template (str): Prompt template for enrichment prompt 
    """
    with open(enrichment_template_path, 'r') as f:
        enrichment_prompt_template = f.read()

    return enrichment_prompt_template


def question_enrichment_few_shot_prep(few_shot_data_path: str, q_id: int, q_db_id: str, level_shot_number: str, schema_existance: bool, enrichment_level: str, mode: str) -> str:
    """
    The function selects the given number of exemple from the question enrichment few-shot data, and then
    concatenate the selected exemples in string format. 
    - The few-shot examples will be selected from the set of data contains databases different than the considered question.
    - If the question is already annotated, then few-shot examples will be selected from the set of data in which given question is excluded.
    - Level shot number can be between 0 to 10.

    Arguments:
        few_shot_data_path (str); path to the file in which few_shot_data exist
        q_id (int): id of the question
        q_db_id (str): database ID (database name)
        level_shot_number (int): number of exemple desired to add in the prompt for each level. 
        schema_existance (bool): Whether the schema will be provided for the exemplars in the prompt. If it is True, then schema will be provided.
        enrichment_level (str): Either "basic" or "complex" for selecting enriched questions
        mode (str): dev mode or test mode
    Returns:
        few_shot_exemplars (str): selected and concatenated exemplars for the prompt
    """
    bird_sql_path = os.getenv('BIRD_DB_PATH')

    if level_shot_number == 0:
        return ""
    
    few_shot_exemplars = ""
    # Check level_shot_number
    if level_shot_number < 0 or level_shot_number > 10:
        raise ValueError("Invalid few-shot number. The level_shot_number should be between 0 and 10")
    
    # Check schema_existance
    if not isinstance(schema_existance, bool): 
        raise ValueError("Invalid value for schema_existance variable,it is not a boolean. It should be either True or False.")
    
    # Check enrichment_level and set enrichment_label
    if enrichment_level == "basic":
        enrichment_label = "question_enriched"
    elif enrichment_level == "complex":
        enrichment_label = "question_enriched_v2"
    else:
        raise ValueError("Invalid value for enrichment_level. The variable must be either 'basic' or 'complex'.")
    
    # Check mode
    if mode not in ['test', 'dev']:
        raise ValueError("Invalid value for mode. The variable must be either 'dev' or 'test'.")

    
    # Get all few-shot exemples
    all_few_shot_data = load_few_shot_data(few_shot_data_path=few_shot_data_path)
    # Set difficulty levels
    levels = ['simple', 'moderate', 'challanging']
   
    
    for level in levels:
        examples_in_level = all_few_shot_data[level]
        selected_indexes = []
        if mode == "dev":
            # remove the annotated questions if their db_id is the same with the considered question's db_id
            examples_in_level_tmp = []
            for example in examples_in_level:
                if q_db_id != example['db_id']:
                    examples_in_level_tmp.append(example)
            examples_in_level = examples_in_level_tmp 
            # By removing the db_ids that is same with the considered question's db_id, the selection of the same question as an example is prevented in the case of the question was in the annotated data
        
        selected_indexes = random.sample(range(0,len(examples_in_level)), level_shot_number) # randomly select exemple_num_for_each_level number of example
        for ind in selected_indexes:
            current_question_info_dict = examples_in_level[ind]

            if schema_existance:
                curr_q_db_id = current_question_info_dict['db_id']
                db_path = bird_sql_path + f"/{mode}/{mode}_databases/{curr_q_db_id}/{curr_q_db_id}.sqlite"
                schema = get_schema(db_path)
                few_shot_exemplars = few_shot_exemplars + "Database Schema: \n" +  schema + '\n'
            
            few_shot_exemplars = few_shot_exemplars + "Question: " + current_question_info_dict['question'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Evidence: " + current_question_info_dict['evidence'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Enrichment Reasoning: " + current_question_info_dict['enrichment_reasoning'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Enriched Question: " + current_question_info_dict[enrichment_label] + "\n\n"

    return few_shot_exemplars


def fill_question_enrichment_prompt_template(template: str, schema: str, db_samples: str, question: str, possible_conditions: str, few_shot_examples: str, evidence: str, db_descriptions: str):
    """
    The functions completes the enrichment prompt template by filling the necessary slots which are schema, db_samples, question, possible_conditions, few_shot_examples, evidence and db_desctiptions 

    Arguments:
        template (str): The template that is going to be filled
        schema (str): The schema of the database to which considered question belong
        questoin (str): The considered question that is going to be enriched
        possible_conditions (str): Possible conditions extracted from the possible SQLite SQL query for the question
        few_shot_examples (str): few-shot examples that are injected to the prompt
        evidence (str): Given evidence statment if exist
        db_descriptions (str): Question relevant database item(column) descriptions

    Returns:
        prompt (str): Completed prompt for question enrichment
    """
    if evidence == '' or evidence == None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"

    if few_shot_examples == '' or few_shot_examples == None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"

    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    db_samples = "\n### Database Samples: \n\n" + db_samples
    question = "\n### Question: \n" + question

    if possible_conditions:
        possible_conditions = "\n### Possible SQL Conditions: \n" + possible_conditions
    else:
        possible_conditions = "\n### Possible SQL Conditions: No strict conditions were found. Please consider the database schema and keywords while enriching the Question."
    

    prompt = template.format(
        FEWSHOT_EXAMPLES = few_shot_examples,
        SCHEMA = schema,
        DB_SAMPLES = db_samples,
        QUESTION = question,
        EVIDENCE = evidence,
        DB_DESCRIPTIONS = db_descriptions,
        POSSIBLE_CONDITIONS = possible_conditions
    )

    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt



def fill_refinement_prompt_template(template: str, schema: str, possible_conditions: str, question: str, possible_sql: str, exec_err: str, few_shot_examples: str = "", evidence: str = "", db_descriptions: str = "") -> str:
    """
    The functions completes the prompt template by filling the necessary slots which are few_shot_examples, schema, question, evidence and db_decriptions, possible_sql, execution error and possible_conditions

    Arguments:
        template (str): The template that is going to be filled
        schema (str): The schema of the database to which considered question belong
        questoin (str): The considered question that is going to be enriched
        possible_sql (str): Possible SQLite SQL query for the question
        exec_err (str): Taken execution error when possible SQL is executed
        few_shot_examples (str): few-shot examples that are injected to the prompt
        evidence (str): Given evidence statment if exist
        db_descriptions (str): Question relevant database item(column) descriptions

    Returns:
        prompt (str): Completed prompt for question enrichment
    """
    if evidence == '' or evidence == None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"

    if few_shot_examples == '' or few_shot_examples == None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"

    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    question = "\n### Question: \n" + question
    possible_sql = "\n### Possible SQLite SQL Query: \n" + possible_sql
    if possible_conditions:
        possible_conditions = "\n### Possible SQL Conditions: \n" + possible_conditions
    else:
        possible_conditions = "\n### Possible SQL Conditions: No strict conditions were found. Please consider the database schema and keywords in the question while generating the SQL."
    if exec_err:
        exec_err = "\n### Execution Error of Possible SQL Query Above: \n" + exec_err + "\n While generating new SQLite SQL query, consider this execution error and make sure newly generated SQL query runs without execution error."
    else:
        exec_err = ""
    

    prompt = template.format(
        FEWSHOT_EXAMPLES = few_shot_examples,
        SCHEMA = schema,
        QUESTION = question,
        EVIDENCE = evidence,
        DB_DESCRIPTIONS = db_descriptions,
        POSSIBLE_SQL_Query = possible_sql,
        EXECUTION_ERROR = exec_err,
        POSSIBLE_CONDITIONS = possible_conditions
    )

    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt


def schema_filtering_few_shot_prep(few_shot_data_path: str, q_db_id: str, level_shot_number:str, schema_existance: bool, mode: str) -> str:
    """
    The function selects the given number of exemple from the few-shot data, and determines the filtered schema of questions in few-shot data and then concatenate the selected exemples in string format. 
    - The few-shot examples will be selected from the set of data contains databases different than the considered question.
    - If the question is already annotated, then few-shot examples will be selected from the set of data in which given question is excluded.
    - Level shot number can be between 0 to 10.

    Arguments:
        few_shot_data_path (str): few-shot data path
        q_db_id (str): database ID (database name) of considered question
        level_shot_number (int): number of exemple desired to add in the prompt for each level. 
        schema_existance (bool): Whether the schema will be provided for the exemplars in the prompt. If it is True, then schema will be provided.
        mode (str): dev mode or test mode
    Returns:
        few_shot_exemplars (str): selected and concatenated exemplars for the prompt
    """
    bird_sql_path = os.getenv('BIRD_DB_PATH', '../../dataset/bird-sql')
    
    if level_shot_number == 0:
        return ""
    
    few_shot_exemplars = ""
    # Check level_shot_number
    if level_shot_number < 0 or level_shot_number > 10:
        raise ValueError("Invalid few-shot number. The level_shot_number should be between 0 and 10")
    
    # Check schema_existance
    if not isinstance(schema_existance, bool): 
        raise TypeError("Provided variable is not a boolean.")
    
    # Check mode
    if mode not in ['test', 'dev']:
        raise ValueError("Invalid value for mode. The variable must be either 'dev' or 'test'.")

    # Get all few-shot exemples
    all_few_shot_data = load_few_shot_data(few_shot_data_path=few_shot_data_path)
    # Set difficulty levels
    levels = ['simple', 'moderate', 'challanging']
   
    for level in levels:
        examples_in_level = all_few_shot_data[level]
        selected_indexes = []
        if mode == "dev":
            # remove the annotated questions if their db_id is the same with the considered question's db_id
            examples_in_level_tmp = []
            for example in examples_in_level:
                if q_db_id != example['db_id']:
                    examples_in_level_tmp.append(example)
            examples_in_level = examples_in_level_tmp 
            # By removing the db_ids that is same with the considered question's db_id, the selection of the same question as an example is prevented in the case of the question was in the annotated data
        
        selected_indexes = random.sample(range(0,len(examples_in_level)), level_shot_number) # randomly select exemple_num_for_each_level number of example
        for ind in selected_indexes:
            current_question_info_dict = examples_in_level[ind]
            curr_q_db_id = current_question_info_dict['db_id']
            db_path = bird_sql_path + f"/{mode}/{mode}_databases/{curr_q_db_id}/{curr_q_db_id}.sqlite"
            sql = current_question_info_dict['SQL']
            

            if schema_existance:
                schema = get_schema(db_path)
                few_shot_exemplars = few_shot_exemplars + "Database Schema: \n" +  schema + '\n'
            
            # print("curent directory", os.getcwd())
            few_shot_exemplars = few_shot_exemplars + "Question: " + current_question_info_dict['question'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Evidence: " + current_question_info_dict['evidence'] + "\n"
            filtered_schema = extract_sql_columns(db_path, sql)
            few_shot_exemplars = few_shot_exemplars + "Filtered Database Schema: \n" + str(filtered_schema) + "\n"

    return few_shot_exemplars


def fill_prompt_template(template: str, schema: str, db_samples: str, question: str, few_shot_examples: str = "", evidence: str = "", db_descriptions: str = "") -> str:
    """
    The functions completes the prompt template by filling the necessary slots which are few_shot_examples, schema, question and evidence 

    Arguments:
        template (str): The template that is going to be filled
        schema (str): The schema of the database to which considered question belong
        questoin (str): The considered question that is going to be enriched
        few_shot_examples (str): few-shot examples that are injected to the prompt
        evidence (str): Given evidence statment if exist
        db_descriptions (str): Question relevant database item(column) descriptions

    Returns:
        prompt (str): Completed prompt for question enrichment
    """
    if evidence == '' or evidence == None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"

    if few_shot_examples == '' or few_shot_examples == None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"

    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    db_samples = "\n### Database Samples: \n\n" + db_samples
    question = "\n### Question: \n" + question

    prompt = template.format(
        FEWSHOT_EXAMPLES = few_shot_examples,
        SCHEMA = schema,
        DB_SAMPLES = db_samples,
        QUESTION = question,
        EVIDENCE = evidence,
        DB_DESCRIPTIONS = db_descriptions
    )

    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    # print("The prompt: \n", prompt)
    return prompt