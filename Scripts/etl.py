import argparse
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, to_date, year, month, weekofyear, dayofweek
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
import pyspark.sql.functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_nulls(df):
    return df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

def fill_missing_with_median(df, numeric_cols):
    for col_name in numeric_cols:
        if col_name in df.columns:
            median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
            if median_val is not None:
                df = df.fillna({col_name: median_val})
    return df

def fill_missing_with_mode(df, categorical_cols):
    for col_name in categorical_cols:
        if col_name in df.columns:
            mode_val = df.groupBy(col_name)\
                         .count()\
                         .orderBy('count', ascending=False)\
                         .first()[0]
            df = df.fillna({col_name: mode_val})
    return df

def join_all(base_df, stores_df, features_df, df_type="train"):
    df = base_df.join(stores_df, on="Store", how="left") \
                .join(features_df, on=["Store", "Date"], how="left")

    numeric_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
                    "CPI", "Unemployment", "Fuel_Price", "Temperature"]
    categorical_cols = ["IsHoliday", "Type"]

    df = fill_missing_with_median(df, numeric_cols)
    df = fill_missing_with_mode(df, categorical_cols)

    if df_type == "train":
        df = df.na.drop(subset=["Weekly_Sales"])

    # Add date-based features
    df = df.withColumn("Year", year("Date")) \
           .withColumn("Month", month("Date")) \
           .withColumn("WeekOfYear", weekofyear("Date")) \
           .withColumn("DayOfWeek", dayofweek("Date"))  # Sunday=1..Saturday=7

    return df

def add_lag_rolling_features(df):
    # Only add lag/rolling if Weekly_Sales column exists (i.e., train data)
    if "Weekly_Sales" not in df.columns:
        return df

    window_spec = Window.partitionBy("Store", "Dept").orderBy("Date")

    df = df.withColumn("Weekly_Sales_lag1", F.lag("Weekly_Sales", 1).over(window_spec))
    df = df.withColumn("Weekly_Sales_lag4", F.lag("Weekly_Sales", 4).over(window_spec))
    df = df.withColumn("Weekly_Sales_roll4", F.avg("Weekly_Sales").over(window_spec.rowsBetween(-4, -1)))

    df = df.fillna({"Weekly_Sales_lag1": 0, "Weekly_Sales_lag4": 0, "Weekly_Sales_roll4": 0})

    return df

def main():
    parser = argparse.ArgumentParser(description="ETL script for Walmart sales data")
    parser.add_argument('--raw_dir', type=str, required=True, help="Path to raw data directory")
    parser.add_argument('--processed_dir', type=str, required=True, help="Path to output processed directory")
    args = parser.parse_args()

    RAW_DIR = args.raw_dir
    PROCESSED_DIR = args.processed_dir

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    spark = SparkSession.builder.appName("Walmart ETL").getOrCreate()

    try:
        logger.info("Loading raw data...")
        train_df = spark.read.csv(f"{RAW_DIR}/train.csv", header=True, inferSchema=True)
        test_df = spark.read.csv(f"{RAW_DIR}/test.csv", header=True, inferSchema=True)
        features_df = spark.read.csv(f"{RAW_DIR}/features.csv", header=True, inferSchema=True)
        stores_df = spark.read.csv(f"{RAW_DIR}/stores.csv", header=True, inferSchema=True)

        # Cast Date columns
        train_df = train_df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
        test_df = test_df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
        features_df = features_df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))

        numeric_cols_to_cast = [
            "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
            "CPI", "Unemployment"
        ]

        for col_name in numeric_cols_to_cast:
            if col_name in features_df.columns:
                features_df = features_df.withColumn(
                    col_name,
                    F.when(col(col_name) == "NA", None).otherwise(col(col_name)).cast(FloatType())
                )

        # Drop 'IsHoliday' from features_df to avoid join ambiguity
        features_df = features_df.drop("IsHoliday")

        # Join, impute, and feature engineer
        train_final = join_all(train_df, stores_df, features_df, df_type="train")
        test_final = join_all(test_df, stores_df, features_df, df_type="test")

        # Add lag and rolling features only to train data
        train_final = add_lag_rolling_features(train_final)

        # Save processed data
        train_output_path = os.path.join(PROCESSED_DIR, "merged_train.parquet")
        test_output_path = os.path.join(PROCESSED_DIR, "merged_test.parquet")

        logger.info(f"Saving processed train data to {train_output_path}")
        train_final.coalesce(1).write.mode("overwrite").parquet(train_output_path)

        logger.info(f"Saving processed test data to {test_output_path}")
        test_final.coalesce(1).write.mode("overwrite").parquet(test_output_path)

        logger.info("ETL processing completed successfully.")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
