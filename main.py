import os
import argparse
import json
import random
from dotenv import load_dotenv
from pipeline.Pipeline import *
from utils.db_utils import * 
from utils.retrieval_utils import process_all_dbs
from typing import Dict, Union, List, Tuple

def main(args):
    load_dotenv() # load variables into os.environ
    create_result_files(args) # creating results directory for specific arguments

    bird_sql_path = os.getenv('BIRD_DB_PATH')
    args.dataset_path = bird_sql_path
    process_all_dbs(bird_sql_path, args.mode) # for all databases, creating db_description.csv files which include column descriptions for all talbes

    # set random seed
    random.seed(args.seed)
    
    # load dataset
    dataset_json_path = bird_sql_path + f"/{args.mode}/{args.mode}.json"
    f = open(dataset_json_path)
    dataset = json.load(f)

    pipeline = Pipeline(args)

    output_dict = {}
    predictions = []
    # Incase error you can restart the code from the point of error using following lines
    # dataset = dataset[<enter_start_question_id>: <enter_end_question_id>]
    # dataset = dataset[<enter_question_id>:]
    dataset = dataset[1135:]
    for ind,t2s_object in enumerate(dataset):
        q_id = t2s_object["question_id"]
        if pipeline.pipeline_order == "CSG-SR":
            t2s_object_prediction = pipeline.forward_pipeline_CSG_SR(t2s_object)
        elif pipeline.pipeline_order == "CSG-QE-SR":
            t2s_object_prediction = pipeline.forward_pipeline_CSG_QE_SR(t2s_object)
        elif pipeline.pipeline_order == "SF-CSG-QE-SR":
            t2s_object_prediction = pipeline.forward_pipeline_SF_CSG_QE_SR(t2s_object)
        else:
            raise ValueError("Wrong value for pipeline_order argument. It must be either CSG-QE-SR or CSG-SR.")
        
        # Compare predicted and ground truth sqls
        compare_results = check_correctness(t2s_object_prediction, args)
        t2s_object_prediction['results'] = compare_results
        if os.path.exists(args.prediction_json_path):
            # get existing predictions
            with open(args.prediction_json_path, 'r') as file_read:
                existing_predictions = json.load(file_read)

            # add new prediction to the existing predictions and then write to the file
            existing_predictions.append(t2s_object_prediction)
            with open(args.prediction_json_path, 'w') as file_write:
                json.dump(existing_predictions, file_write, indent=4)

        else:
            file_write = open(args.prediction_json_path, 'w')
            existing_predictions = [t2s_object_prediction]
            json.dump(existing_predictions, file_write, indent=4)
            file_write.close()

        # # add the current text2sql object to the predictions
        # predictions.append(t2s_object_prediction)
        # # writing prediction to the predictions.json file
        # with open(args.prediction_json_path, 'w') as f:
        #     json.dump(predictions, f, indent=4)  # indent=4 for pretty printing

        # adding predicted sql in the expected format for the evaluation files
        db_id = t2s_object_prediction["db_id"]
        predicted_sql = t2s_object_prediction["predicted_sql"]
        predicted_sql = predicted_sql.replace('\"','').replace('\\\n',' ').replace('\n',' ')
        sql = predicted_sql + '\t----- bird -----\t' + db_id
        output_dict[str(q_id)] = sql
        if os.path.exists(args.predictions_eval_json_path):
            with open(args.predictions_eval_json_path, 'r') as f:
                contents = json.loads(f.read())
        else:
            # Initialize contents as an empty dictionary if the file doesn't exist
            contents = {}
        contents.update(output_dict)
        json.dump(contents, open(args.predictions_eval_json_path, 'w'), indent=4)
        
        print(f"Question with {q_id} is processed. Correctness: {compare_results['exec_res']} ")

    # Calculatin Metrics
    predictions_json_file = open(args.prediction_json_path, 'r')
    predictions = json.load(predictions_json_file)
    stats, fail_q_ids = calculate_accuracies(predictions)
    metric_object = {
        "EX": stats["ex"],
        "total_correct_count": stats["total_correct_count"],
        "total_item_count": stats["total_item_count"],
        "simple_stats": stats["simple"],
        "moderate_stats": stats["moderate"],
        "challenging_stats": stats["challenging"],
        "fail_q_ids": fail_q_ids,
        "config": {
            "mode": args.mode,
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
            "pipeline_order": args.pipeline_order,
            "enrichment_level": args.enrichment_level,
            "enrichment_level_shot_number": args.enrichment_level_shot_number,
            "enrichment_few_shot_schema_existance": args.enrichment_few_shot_schema_existance,
            "filtering_level_shot_number": args.filtering_level_shot_number,
            "filtering_few_shot_schema_existance": args.filtering_few_shot_schema_existance,
            "cfg": args.cfg,
            "generation_level_shot_number": args.generation_level_shot_number,
            "generation_few_shot_schema_existance": args.generation_few_shot_schema_existance,
            "db_sample_limit": args.db_sample_limit,
            "relevant_description_number": args.relevant_description_number,
            "seed": args.seed
        }
    }

    # Writing metrics to a file
    metrics_path = args.output_directory_path + "/metrics.json"
    # writing metric object 
    with open(metrics_path, 'w') as f:
        json.dump(metric_object, f, indent=4) # indent=4 for pretty printing

    print("Metrics are written into metrics.json file.")
    return 


def calculate_accuracies(predictions: List[Dict]) -> Tuple[float, List]:
    """
    The function calculates the Execution Accuracy(EX) metric and find the question IDs whose predictions are failed

    Arguments:
        predictions
    """
    difficulty_flag = False
    failed_predictions_q_ids = []
    stats = {
        "ex": 0,
        "total_correct_count": 0,
        "total_item_count": 0,
        "simple": {
            "correct_number": 0,
            "count": 0
        },
        "moderate": {
            "correct_number": 0,
            "count": 0
        },
        "challenging": {
            "correct_number": 0,
            "count": 0
        }
    }

    # check if there is difficulty key
    sample = predictions[0]
    if "difficulty" in sample:
        difficulty_flag = True
    else:
        difficulty_flag = False

    if difficulty_flag:
        for q2s_object in predictions:
            level = q2s_object['difficulty']
            stats[level]["count"] = stats[level]["count"] + 1

            if q2s_object['results']['exec_res'] != 0:
                stats[level]['correct_number'] = stats[level]['correct_number'] + 1
            else:
                failed_predictions_q_ids.append(q2s_object['question_id'])

        stats["simple"]["ex"] = stats["simple"]["correct_number"] / stats["simple"]["count"] * 100
        stats["moderate"]["ex"] = stats["moderate"]["correct_number"] / stats["moderate"]["count"] * 100
        stats["challenging"]["ex"] = stats["challenging"]["correct_number"] / stats["challenging"]["count"] * 100

        stats["total_item_count"] = stats["simple"]["count"] + stats["moderate"]["count"] + stats["challenging"]["count"]
        stats["total_correct_count"] = stats["simple"]["correct_number"] + stats["moderate"]["correct_number"] + stats["challenging"]["correct_number"]
        stats["ex"] = stats["total_correct_count"] / stats["total_item_count"] * 100

        return (stats, failed_predictions_q_ids)
    
    else:
        
        for q2s_object in predictions:
            stats["total_item_count"] = stats["total_item_count"]  + 1
            if q2s_object['results']['exec_res'] != 0:
                stats["total_correct_count"] = stats["total_correct_count"]  + 1
            else:
                failed_predictions_q_ids.append(q2s_object['question_id'])

        stats["ex"] = stats["total_correct_count"] / stats["total_item_count"] * 100
        return (stats, failed_predictions_q_ids)

def check_correctness(t2s_object_prediction: Dict, args) -> Dict[str, Union[int, str]]:
    """
    The function check whether predicted SQL is correct or not

    Arguments:
        t2s_object_prediction ()

    Returns:
        compare_results (Dict[str, Union[int, str]]): Comparison results dictionary with execution result and execution error keys
    """
    db_id = t2s_object_prediction['db_id']
    bird_sql_path = os.getenv('BIRD_DB_PATH')
    db_path = bird_sql_path+ f"/{args.mode}/{args.mode}_databases/{db_id}/{db_id}.sqlite"
    if 'predicted_sql' in t2s_object_prediction:
        predicted_sql = t2s_object_prediction['predicted_sql']
        gt_sql = t2s_object_prediction['SQL']
        compare_results = compare_sqls(db_path=db_path, predicted_sql=predicted_sql, ground_truth_sql=gt_sql )
    else:
        compare_results = {'exec_res': 0, 'exec_err': "There is no predicted SQL. There must be and error in this question while extracting information."}

    return compare_results

def create_result_files(args):
    """
    The function creates result files according to arguments.
    """

    # Ensure the results directory exist otherwise create it
    if not os.path.exists("./results"):
        os.makedirs("./results")

    args.output_directory_path = f"./results/model_outputs_{args.mode}_{args.pipeline_order}_{args.model}"

    # Ensure the directory exists
    if not os.path.exists(args.output_directory_path):
        os.makedirs(args.output_directory_path)

    # Overall predictions file
    prediction_json_path = args.output_directory_path + "/predictions.json"
    args.prediction_json_path = prediction_json_path
    # print("args.prediction_json_path: ", args.prediction_json_path)

    # Create an empty predictions.json file if not exist
    if not os.path.exists(prediction_json_path):
        with open(args.prediction_json_path, 'w') as f:
            json.dump([], f)  # Initialize with an empty JSON object

    # predictions file for evaluation
    predictions_eval_json_path = args.output_directory_path + f"/predict_{args.mode}.json"
    args.predictions_eval_json_path = predictions_eval_json_path



def str2bool(v: str) -> bool:
    """
    The function converst string boolean to boolean
    
    Arguments:
        v (str): string boolean
    
    Returns:
        Bool: corresponding boolean variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Running mode arguments
    parser.add_argument("--mode", default='dev', type=str, help="Either dev or test.")

    # Model Arguments
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", type=str, help="OpenAI models.")
    parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature between 0 to 2. It is recommended altering this or top_p but not both.")
    parser.add_argument("--top_p", default=1, type=float, help="Nucleus sampling. It is recommend altering this or temperature but not both")
    parser.add_argument("--max_tokens", default=2048, type=int, help="The maximum number of tokens that can be generated.")
    parser.add_argument("--n", default=1, type=int, help="How many chat completion choices to generate for each input message")

    # Pipeline Arguments
    parser.add_argument("-po", "--pipeline_order", default='EFG', type=str, help="The order of stages in the pipeline. It should be either EFG (enrichment --> filtering --> generation) or FEG (filtering --> enrichment --> generation)")

    # Question Enrichment Arguments
    parser.add_argument("-el", "--enrichment_level", default="complex", type=str, help="Defines the which enrichment is used in few-shot examples.It can be either basic or complex.")
    parser.add_argument("-elsn", "--enrichment_level_shot_number", default=3, type=int, help="The few-shot number for each difficulty level for question enrichment stage.")
    parser.add_argument("-efsse", "--enrichment_few_shot_schema_existance", default=False, type=str2bool, help="Database Schema usage for each few-shot examples in the question enrichment stage. Default False.")

    # Schema Filtering Arguments
    parser.add_argument("-flsn", "--filtering_level_shot_number", default=3, type=int, help="The few-shot number for each difficulty level for schema filtering stage.")
    parser.add_argument("-ffsse", "--filtering_few_shot_schema_existance", default=False, type=str2bool, help="Database Schema usage for each few-shot examples in the schema filtering stage. Default False.")

    # SQL Generation Arguments
    parser.add_argument("--cfg", default=True, type=str2bool, help="Whether Context-Free-Grammer or SQL Template will be used. Default is True.")
    parser.add_argument("-glsn", "--generation_level_shot_number", default=3, type=int, help="The few-shot number for each difficulty level for SQL generation stage.")
    parser.add_argument("-gfsse", "--generation_few_shot_schema_existance", default=False, type=str2bool, help="Database Schema usage for each few-shot examples in the SQL generation stage. Default False.")

    # db sample number
    parser.add_argument("--db_sample_limit", default=5, type=int, help="The number of value extracted for a column for database samples.")
    # question relevant database item/column description number
    parser.add_argument("-rdn", "--relevant_description_number", default=6, type=int, help="The number of database item/column descriptions added to a prompt.")
    # custom seed argument
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    args = parser.parse_args()
    main(args)