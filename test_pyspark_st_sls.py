from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, lit, isnull

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("test_transformation") \
    .getOrCreate()

# Dummy Data for CFACIL
cfacil_data = [
    ("FAC1", "Facility 1", "DIV1", 10.123456, 20.123456), # Normal match
    ("FAC2", "Facility 2", "", 10.0, 20.0),             # Empty DIVI (should become "0" and get KEY 0)
    ("FAC3", "Facility 3", None, 10.0, 20.0),           # Null DIVI (should become "0" and get KEY 0)
    ("FAC4", "Facility 4", "DIV_X", 10.0, 20.0),        # Unmatched DIVI (should get KEY 777)
    ("FAC5", "Facility 5", "DIV1 ", 10.0, 20.0),        # Trailing space match test
]
cfacil_df = spark.createDataFrame(cfacil_data, ["FACI", "FACN", "DIVI", "GEOX", "GEOY"])

# Dummy Data for ProfitCentreDMS
pft_data = [
    (100, "DIV1"),
    (200, "DIV2"),
]
profit_centre_dms_df = spark.createDataFrame(pft_data, ["PFT_CTR_DMS_KEY", "PFT_CTR_CD"])

azure_extract_id = "EXT-12345"

print("--- Source CFACIL ---")
cfacil_df.show()

# Transformation: derivedColumn1
derived_column1_df = cfacil_df.withColumn(
    "DIVI",
    when(isnull(col("DIVI")) | (trim(col("DIVI")) == ""), "0").otherwise(col("DIVI"))
)

# Transformation: join1
join1_df = derived_column1_df.join(
    profit_centre_dms_df,
    trim(derived_column1_df["DIVI"]) == trim(profit_centre_dms_df["PFT_CTR_CD"]),
    "left"
)

# Transformation: derivedColumn2
derived_column2_df = join1_df.withColumn(
    "PFT_CTR_DMS_KEY",
    when(derived_column1_df["DIVI"] == "0", 0)
    .when(derived_column1_df["DIVI"] == profit_centre_dms_df["PFT_CTR_CD"], profit_centre_dms_df["PFT_CTR_DMS_KEY"])
    .when(trim(derived_column1_df["DIVI"]) == trim(profit_centre_dms_df["PFT_CTR_CD"]), profit_centre_dms_df["PFT_CTR_DMS_KEY"]) # Handling trimmed match just like in join
    .otherwise(777)
).withColumn(
    "AZ_EXT_ID",
    lit(azure_extract_id).cast("string")
)

# Sink mapping
sink_mapped_df = derived_column2_df.select(
    col("FACI").alias("FCY_CD"),
    col("FACN").alias("FCY_DSC"),
    col("PFT_CTR_DMS_KEY"),
    col("GEOX").cast("decimal(21,6)").alias("GEO_LTD"),
    col("GEOY").cast("decimal(21,6)").alias("GEO_LGT"),
    col("AZ_EXT_ID")
)

print("--- Final Sink DataFrame ---")
sink_mapped_df.show()
