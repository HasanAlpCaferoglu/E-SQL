from openai import OpenAI
from typing import Dict

def create_response(stage: str, prompt: str, model: str, max_tokens: int, temperature: float, top_p: float,  n: int) -> Dict:
    """
    The functions creates chat response by using chat completion

    Arguments:
        stage (str): stage in the pipeline
        prompt (str): prepared prompt 
        model (str): LLM model used to create chat completion
        max_tokens (int): The maximum number of tokens that can be generated in the chat completion
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling
        n (int): Number of chat completion for each input message

    Returns:
        response_object (Dict): Object returned by the model
    """
    client = OpenAI()

    if stage == "question_enrichment":
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
    elif stage == "candidate_sql_generation":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "sql_refinement":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "schema_filtering":
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
    else:
        raise ValueError("Wrong value for stage. It can only take following values: question_enrichment, candidate_sql_generation, sql_refinement or schema_filtering.")

    response_object = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        max_tokens = max_tokens,
        response_format = { "type": "json_object" },
        temperature = temperature,
        top_p = top_p,
        n=n,
        presence_penalty = 0.0,
        frequency_penalty = 0.0
    )

    return response_object


def upload_file_to_openai(file_path: str) -> Dict:
    """
    The function uploads given file to opanai for batch processing.

    Arguments:
        file_path (str): path of the file that is going to be uplaoded
    Returns:
        file_object (FileObject): Returned file object by openai
    """
    client = OpenAI()

    file_object = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    print("File is uploaded to OpenAI")
    return file_object


def construct_request_input_object(prompt: str, id: int, model: str, system_message: str) -> Dict:
    """
    The function creates a request input object for each item in the dataset

    Arguments:
        prompt (str): prompt that is going to given to the LLM as content
        id (int); the id of the request
        model (str): LLM model name
        system_message (str): the content of the system message

    Returns:
        request_input_object (Dict): The dictionary format required to be for request input
    """
    request_input_object = {
        "custom_id": f"qe-request-{id}",
        "method": "POST",
        "url": "/v1/chat/completions", 
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system_message}"}, 
                {"role": "user", "content": f"{prompt}"}
                ]
        }
    }
    return request_input_object