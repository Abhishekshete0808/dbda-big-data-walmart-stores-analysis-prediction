from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Abhishek',                 
    'start_date': datetime(2025, 8, 6),  
    'retries': 1,                        
    'retry_delay': timedelta(minutes=5)  
}

with DAG(
    'walmart_manual_pipeline',            
    description='Manual DAG to run ETL, EDA, and Model scripts sequentially',
    default_args=default_args,
    schedule_interval=None,  # Run manually only
    catchup=False,                      
    max_active_runs=1,              
) as dag:

    run_etl = BashOperator(
        task_id='run_etl',
        bash_command=(
            "source /home/abhishek/myenv/bin/activate && "
            "python3 /home/abhishek/DBDA/Walmart_Sales_Prediction/Scripts/etl.py "
            "--raw_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Data/Raw "
            "--processed_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Data/Processed"
        ),
    )

    run_eda = BashOperator(
        task_id='run_eda',
        bash_command=(
            "source /home/abhishek/myenv/bin/activate && "
            "python3 /home/abhishek/DBDA/Walmart_Sales_Prediction/Scripts/eda.py "
            "--processed_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Data/Processed/ "
            "--plots_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Plots"
        ),
    )

    run_model = BashOperator(
        task_id='run_model',
        bash_command=(
            "source /home/abhishek/myenv/bin/activate && "
            "python3 /home/abhishek/DBDA/Walmart_Sales_Prediction/Scripts/model.py "
            "--processed_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Data/Processed/ "
            "--output_dir /home/abhishek/DBDA/Walmart_Sales_Prediction/Output"
        ),
    )

    run_etl >> run_eda >> run_model
