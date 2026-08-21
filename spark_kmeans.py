from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

#start spark
spark = SparkSession.builder \
    .appName("CloudWatch_Traffic_KMeans") \
    .enableHiveSupport() \
    .getOrCreate()

#1. fetch data from Hive table
df = spark.sql("SELECT bytes_in, bytes_out FROM cloudwatch_web_attacks")

#2. assemble features into a vector column
assembler = VectorAssembler(
    inputCols = ["bytes_in", "bytes_out"], 
    outputCol = "raw_features"
)
data = assembler.transform(df)

#3. standardize features (mean = 0, std = 1)
scaler = StandardScaler(
    inputCol = "raw_features", 
    outputCol = "features", 
    withStd = True, 
    withMean = True
)
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)

#4. train K-Means model (k = 3)
kmeans = KMeans(k = 3, seed = 42)
model = kmeans.fit(scaled_data)

#5. make predictions & view cluster assignments
predictions = model.transform(scaled_data)
predictions.select("bytes_in", "bytes_out", "prediction").show(20)

spark.stop()