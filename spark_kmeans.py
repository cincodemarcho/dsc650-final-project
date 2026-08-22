from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import happybase
from datetime import datetime

#start spark
spark = SparkSession.builder \
    .appName("CloudWatch_Traffic_KMeans") \
    .enableHiveSupport() \
    .getOrCreate()

#1. fetch data from Hive table
df = spark.sql("SELECT bytes_in, bytes_out, src_ip, creation_time FROM cloudwatch_web_attacks")
data = df.dropna(subset = ["bytes_in", "bytes_out"])

#2. assemble and scale features
assembler = VectorAssembler(
    inputCols = ["bytes_in", "bytes_out"],
    outputCol = "raw_features",
    handleInvalid = "skip" 
)
data = assembler.transform(data)

scaler = StandardScaler(
    inputCol = "raw_features",
    outputCol = "features",
    withStd = True,
    withMean = True
)
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)

#3. train k-means model (k = 3)
kmeans = KMeans(k = 3, seed = 42)
model = kmeans.fit(scaled_data)

#4. make predictions
predictions = model.transform(scaled_data)

# --- TRAINING OUTPUT ---
print("=========================================")
print("           MODEL TRAINING OUTPUT")
print("=========================================")
print("Cluster Centers (Scaled Features):")
for i, center in enumerate(model.clusterCenters()):
    print(f"  Cluster {i}: {center}")

print("\nSample Predictions:")
predictions.select("bytes_in", "bytes_out", "prediction").show(10)

# --- EVALUATION METRICS ---
print("=========================================")
print("           EVALUATION METRICS")
print("=========================================")

#Metric 1: Silhouette Score (range: -1 to +1, higher is better)
evaluator = ClusteringEvaluator(
    predictionCol="prediction",
    featuresCol = "features",
    metricName = "silhouette"
)
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score with Squared Euclidean Distance = {silhouette:.4f}")

#Metric 2: Within Set Sum of Squared Errors / Cost (WSSE)
wsse = model.summary.trainingCost
print(f"Within Set Sum of Squared Errors (WSSE / Inertia) = {wsse:.4f}")
print("=========================================")

#5. write to HDFS
output_df = predictions.select("bytes_in", "bytes_out", "prediction")
output_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs:///tmp/cloudwatch_kmeans_output")

#6. write metrics and predictions to HBase
connection = happybase.Connection('master', port=9090)
table = connection.table('cloudwatch_web_attacks')

#Summary row with evaluation metrics
b = table.batch()
b.put(b'metrics_summary', {
    b'cf:silhouette_score': str(silhouette).encode(),
    b'cf:wsse': str(wsse).encode(),
    b'cf:k': b'3',
    b'cf:timestamp': str(datetime.now()).encode()
})
b.send()

#per-record predictions
rows = predictions.select("src_ip", "creation_time", "bytes_in", "bytes_out", "prediction").collect()
b = table.batch()
for row in rows:
    row_key = f"{row['src_ip']}_{row['creation_time']}".encode()
    b.put(row_key, {
        b'cf:bytes_in': str(row['bytes_in']).encode(),
        b'cf:bytes_out': str(row['bytes_out']).encode(),
        b'cf:cluster': str(row['prediction']).encode()
    })
b.send()
connection.close()

print("Wrote predictions and metrics to HBase table cloudwatch_web_attacks")

spark.stop()