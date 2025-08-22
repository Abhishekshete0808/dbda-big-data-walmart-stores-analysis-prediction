# dbda-big-data-walmart-stores-analysis-prediction

An end-to-end pipeline for Walmart sales forecasting using PySpark for ETL, Matplotlib/Seaborn for EDA, and a Random Forest model.
The workflow is orchestrated with Apache Airflow, running ETL, EDA, and modeling sequentially.
Visualizations are also created interactively on Databricks Community Edition.

---

## Project Overview

- Data ingestion and cleaning with PySpark

- Exploratory Data Analysis with graphical plots

- Machine Learning model training and evaluation (Random Forest)

- Pipeline orchestration via Airflow DAG

- Interactive visualization demo on Databricks

## Project Structure

````Walmart_Sales_Prediction/
Data/
 ├── Raw/                     # Original datasets  
 └── Processed/               # Cleaned & transformed data  

Scripts/
 ├── etl.py                   # Data processing with PySpark  
 ├── eda.py                   # Exploratory analysis and plotting  
 ├── model.py                 # Model training and prediction  
 ├── airflow_dag.py           # Airflow DAG definition  

Output/
 ├── random_forest_model/     # Saved trained model  
 ├── random_forest_predictions.parquet  
 ├── validation_predictions.parquet  

Plots/                        # EDA generated plots & Databricks notebook link 

requirements.txt              # Python dependencies  
README.md                     # Project documentation  
.gitignore                    # Git ignore rules


----


## Contributors

- Abhishek Shete (https://github.com/Abhishekshete0808)  
- Shivam Chincholkar (https://github.com/Shivam22112)



