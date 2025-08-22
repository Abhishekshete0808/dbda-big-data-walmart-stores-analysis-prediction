import argparse
import os
import logging
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_pipeline(feature_columns, categorical_columns):
    """
    Builds a Spark ML Pipeline consisting of:
    - StringIndexer stages for categorical columns
    - VectorAssembler stage to assemble all features into a single vector
    - RandomForestRegressor as the estimator
    """
    stages = []

    # Index categorical columns (handle invalid entries by keeping them)
    for cat_col in categorical_columns:
        stages.append(StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep"))

    # Separate numeric and indexed categorical columns
    non_cat_cols = [c for c in feature_columns if c not in categorical_columns]
    indexed_cat_cols = [c + "_idx" for c in categorical_columns]

    # Final features include numeric and indexed categorical columns
    final_features = non_cat_cols + indexed_cat_cols

    # Assemble features into a single vector column "features"
    stages.append(VectorAssembler(inputCols=final_features, outputCol="features"))

    # Setup RandomForestRegressor with features and label columns
    rf = RandomForestRegressor(featuresCol="features", labelCol="Weekly_Sales", seed=42)
    stages.append(rf)

    # Create pipeline with all stages
    pipeline = Pipeline(stages=stages)
    return pipeline, rf  # Return both for hyperparameter tuning

def main():
    parser = argparse.ArgumentParser(description="Random Forest Regression with Hyperparameter Tuning for Walmart Sales")
    parser.add_argument('--processed_dir', type=str, required=True, help='Path to processed parquet input data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save model and predictions')
    args = parser.parse_args()

    # Create output directory if missing
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Spark Session
    spark = SparkSession.builder \
    .appName("WalmartSalesRandomForestTuning") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()


    try:
        # Load training and test datasets
        train_df = spark.read.parquet(os.path.join(args.processed_dir, "merged_train.parquet"))
        test_df = spark.read.parquet(os.path.join(args.processed_dir, "merged_test.parquet"))

        # Split training data into training and validation sets (80% train, 20% validation)
        train_data, val_data = train_df.randomSplit([0.8, 0.2], seed=42)

        # Columns to exclude from features
        exclude_cols = ["Weekly_Sales", "Date"]
        # Lag and rolling features created in ETL pipeline and used for training
        new_features = ["Weekly_Sales_lag1", "Weekly_Sales_lag4", "Weekly_Sales_roll4"]

        # Prepare final feature columns: existing features + new lag features
        feature_columns = [c for c in train_df.columns if c not in exclude_cols] + new_features

        # Identify categorical columns among the features
        categorical_columns = [c for c, dtype in train_df.dtypes if dtype == "string" and c in feature_columns]

        logger.info("Building pipeline with features: %s", feature_columns)
        logger.info("Categorical columns identified: %s", categorical_columns)

        # Build pipeline and get the RandomForestRegressor for tuning
        pipeline, rf = build_pipeline(feature_columns, categorical_columns)

        # Define hyperparameter grid for model tuning
        param_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100, 150]) \
            .addGrid(rf.maxDepth, [5, 10, 15]) \
            .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
            .build()

        # Define evaluator metric
        evaluator = RegressionEvaluator(labelCol="Weekly_Sales", predictionCol="prediction", metricName="r2")

        # Setup 3-fold CrossValidator for hyperparameter tuning
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3,
                                  seed=42)

        logger.info("Starting model training and hyperparameter tuning...")
        # Train model with cross-validation
        cv_model = crossval.fit(train_data)

        logger.info("Evaluating tuned model on validation set...")
        # Predict and evaluate on validation data
        val_predictions = cv_model.transform(val_data)
        r2 = evaluator.evaluate(val_predictions)
        logger.info(f"Tuned model validation R²: {r2:.4f}")

        # Save validation set predictions for visualization and further analysis
        val_pred_path = os.path.join(args.output_dir, "validation_predictions.parquet")
        val_predictions.select(*val_data.columns, "prediction").write.mode("overwrite").parquet(val_pred_path)
        logger.info(f"Validation predictions saved to {val_pred_path}")



        # --- Critical step: add missing lag features to test set with zeros ---
        for col_name in new_features:
            if col_name not in test_df.columns:
                # This prevents errors caused by missing columns during prediction
                test_df = test_df.withColumn(col_name, F.lit(0))

        logger.info("Generating predictions on test set...")
        # Generate predictions on test set
        test_predictions = cv_model.transform(test_df)

        # Save predictions to parquet
        pred_path = os.path.join(args.output_dir, "random_forest_predictions.parquet")
        test_predictions.select(*test_df.columns, "prediction").write.mode("overwrite").parquet(pred_path)
        logger.info(f"Predictions saved to {pred_path}")

        # Save the best tuned model
        model_path = os.path.join(args.output_dir, "random_forest_model")
        cv_model.bestModel.write().overwrite().save(model_path)
        logger.info(f"Tuned model saved to {model_path}")

    finally:
        # Always stop SparkSession to free resources
        spark.stop()

if __name__ == "__main__":
    main()
