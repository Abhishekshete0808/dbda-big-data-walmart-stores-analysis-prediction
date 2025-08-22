import argparse
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def count_nulls(df):
    """Count number of nulls per column in a Spark DataFrame."""
    return df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Walmart Sales EDA Script")
    parser.add_argument('--processed_dir', type=str, required=True,
                        help="Path to processed data directory containing parquet files")
    parser.add_argument('--plots_dir', type=str, default="../Plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    processed_dir = args.processed_dir
    plots_dir = args.plots_dir

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    sns.set(style="whitegrid")  # set consistent style for seaborn

    # Initialize SparkSession
    spark = SparkSession.builder.appName("Walmart Sales EDA").getOrCreate()

    # Load processed parquet data
    train_path = os.path.join(processed_dir, "merged_train.parquet")
    test_path = os.path.join(processed_dir, "merged_test.parquet")

    logger.info("Loading processed train data...")
    train_df = spark.read.parquet(train_path)

    logger.info("Loading processed test data...")
    test_df = spark.read.parquet(test_path)

    # Basic Data Exploration
    logger.info("\nTraining Dataset Overview")
    logger.info(f"Total Rows: {train_df.count()}, Total Columns: {len(train_df.columns)}")
    train_df.printSchema()
    train_df.show(5)

    # Null Value Check
    logger.info("\nNull Counts in Training Data")
    count_nulls(train_df).show()

    # Descriptive Statistics
    logger.info("\nDescriptive Statistics")
    train_df.describe().show()

    # Sampling for visualization (10%)
    sample_fraction = 0.1
    logger.info(f"\nSampling {sample_fraction*100}% of training data for visualizations...")
    sample_pd = train_df.sample(False, sample_fraction, seed=42).toPandas()

    # Distribution of Weekly_Sales
    plt.figure(figsize=(10,6))
    plt.hist(sample_pd["Weekly_Sales"], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Weekly Sales")
    plt.xlabel("Weekly Sales")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "weekly_sales_distribution.png"))
    plt.show()

    # Outlier Detection (IQR method)
    Q1 = sample_pd["Weekly_Sales"].quantile(0.25)
    Q3 = sample_pd["Weekly_Sales"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    logger.info(f"\nWeekly Sales Outlier Thresholds:\nLower Bound = {lower_bound}\nUpper Bound = {upper_bound}")
    outlier_count = sample_pd[(sample_pd["Weekly_Sales"] < lower_bound) | (sample_pd["Weekly_Sales"] > upper_bound)].shape[0]
    logger.info(f"Number of potential outlier records: {outlier_count}")

    # Correlation: Numeric Features vs Weekly Sales
    correlation = sample_pd.corr(numeric_only=True)["Weekly_Sales"].sort_values(ascending=False)
    logger.info("\nCorrelation of Numeric Features with Weekly Sales")
    logger.info(f"\n{correlation}")

    plt.figure(figsize=(8,6))
    sns.barplot(x=correlation.index.drop("Weekly_Sales"), y=correlation.drop("Weekly_Sales").values, palette="viridis")
    plt.title("Correlation of Numeric Features with Weekly Sales")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "numeric_feature_correlations.png"))
    plt.show()

    # Time-Based Trends and Seasonality - Monthly Sales Pattern
    plt.figure(figsize=(10,6))
    sns.boxplot(x="Month", y="Weekly_Sales", data=sample_pd)
    plt.title("Monthly Distribution of Weekly Sales")
    plt.xlabel("Month")
    plt.ylabel("Weekly Sales")
    plt.savefig(os.path.join(plots_dir, "monthly_sales_boxplot.png"))
    plt.show()

    # Day of Week Sales Pattern
    plt.figure(figsize=(10,6))
    sns.boxplot(x="DayOfWeek", y="Weekly_Sales", data=sample_pd)
    plt.title("Day of Week Distribution of Weekly Sales")
    plt.xlabel("Day of Week (Sunday=1)")
    plt.ylabel("Weekly Sales")
    plt.savefig(os.path.join(plots_dir, "dayofweek_sales_boxplot.png"))
    plt.show()

    # Holiday Impact on Sales
    plt.figure(figsize=(8,5))
    sns.boxplot(x="IsHoliday", y="Weekly_Sales", data=sample_pd)
    plt.title("Impact of Holiday Weeks on Weekly Sales")
    plt.xlabel("IsHoliday")
    plt.ylabel("Weekly Sales")
    plt.savefig(os.path.join(plots_dir, "holiday_sales_boxplot.png"))
    plt.show()

    # Store-Level Insights - Top 10 Stores by Average Weekly Sales
    top_stores = sample_pd.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False).head(10)
    logger.info("\nTop 10 Stores by Average Weekly Sales:")
    logger.info(f"\n{top_stores}")

    plt.figure(figsize=(10,6))
    top_stores.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Top 10 Stores by Average Weekly Sales")
    plt.xlabel("Store")
    plt.ylabel("Average Weekly Sales")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_stores_sales.png"))
    plt.show()

    # Store Type Impact on Sales
    if "Type" in sample_pd.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(x="Type", y="Weekly_Sales", data=sample_pd)
        plt.title("Weekly Sales Distribution by Store Type")
        plt.xlabel("Store Type")
        plt.ylabel("Weekly Sales")
        plt.savefig(os.path.join(plots_dir, "store_type_sales_boxplot.png"))
        plt.show()

    logger.info("\nEDA completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
