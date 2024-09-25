db_root_path='../dataset/bird-sql/dev/dev_databases/'
data_mode='dev' # dev, train, mini_dev
diff_json_path='../dataset/bird-sql/dev/dev.json' # _sqlite.json, _mysql.json, _postgresql.json
# Path where the predicted SQL queries are stored
predicted_sql_path='./results/model_outputs_dev_SF-CSG-QE-SR_gpt-4o-mini-2024-07-18/'

ground_truth_path='../dataset/bird-sql/dev/'
num_cpus=72
meta_time_out=60.0
mode_gt='gt'
mode_predict='gpt'

# Choose the engine to run, e.g. gpt-4, gpt-4-32k, gpt-4-turbo, gpt-35-turbo, GPT35-turbo-instruct
engine='gpt-4o'


# Choose the SQL dialect to run, e.g. SQLite, MySQL, PostgreSQL
# PLEASE NOTE: You have to setup the database information in evaluation_utils.py 
# if you want to run the evaluation script using MySQL or PostgreSQL
sql_dialect='SQLite'

echo "starting to compare with knowledge for soft-f1 engine: ${engine} sql_dialect: ${sql_dialect} meta_time_out: ${meta_time_out}"
python3 -u ./evaluation/evaluation_f1.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --engine ${engine} --sql_dialect ${sql_dialect}

echo "starting to compare with knowledge for ex engine: ${engine} sql_dialect: ${sql_dialect} meta_time_out: ${meta_time_out}"
python3 -u ./evaluation/evaluation_ex.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} --engine ${engine} --sql_dialect ${sql_dialect}


echo "starting to compare with knowledge for ves engine: ${engine} sql_dialect: ${sql_dialect} meta_time_out: ${meta_time_out}"
python3 -u ./evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --engine ${engine} --sql_dialect ${sql_dialect}

