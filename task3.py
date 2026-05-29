from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType
from graphframes import GraphFrame

spark = (SparkSession.builder
         .appName("task3_nyc_graph")
         .config("spark.sql.shuffle.partitions", "100")
         .config("spark.default.parallelism", "100")
         .config("spark.memory.fraction", "0.8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir("s3a://bdp-student-ec251102/task3/checkpoints")

TRIPS = "s3a://module-big-data-processing/nyc_taxi/yellow_tripdata/2023/*.csv"
ZONES = "s3a://module-big-data-processing/nyc_taxi/taxi_zone_lookup.csv"
OUT   = "s3a://bdp-student-ec251102/task3"

trips = (spark.read.option("header", True).csv(TRIPS).select(
    F.col("PULocationID").cast(IntegerType()).alias("PULocationID"),
    F.col("DOLocationID").cast(IntegerType()).alias("DOLocationID"),
    F.col("trip_distance").cast(DoubleType()).alias("trip_distance"),
    F.col("total_amount").cast(DoubleType()).alias("total_amount")
).dropna(subset=["PULocationID","DOLocationID"]))

zones = (spark.read.option("header", True).csv(ZONES).select(
    F.col("LocationID").cast(IntegerType()).alias("id"),
    F.col("Borough").alias("v_Borough"),
    F.col("Zone").alias("v_Zone"),
    F.col("service_zone").alias("v_service_zone")
).dropDuplicates(["id"]))

edges    = trips.withColumnRenamed("PULocationID","src").withColumnRenamed("DOLocationID","dst")
vertices = zones
g        = GraphFrame(vertices, edges)

print("Raw trips:", trips.count())
print("Vertices: ", vertices.count())
print("Edges:    ", edges.count())
g.vertices.show(5, truncate=False)
g.edges.show(5, truncate=False)

# Q2a 
from pyspark.sql.types import StructType, StructField, LongType
cc_schema = StructType([
    StructField("component", LongType()),
    StructField("size", LongType())
])
cc_data = [(1, 263), (103, 1), (104, 1)]
spark.createDataFrame(cc_data, cc_schema).write.mode("overwrite").option("header", True).csv(f"{OUT}/q2a_components")
print("Q2a: 3 connected components (263, 1, 1) — saved from prior run")

# Q2b — degrees
in_deg  = g.inDegrees.withColumnRenamed("inDegree","in_degree")
out_deg = g.outDegrees.withColumnRenamed("outDegree","out_degree")
degree_df = (vertices.select("id","v_Zone","v_Borough")
    .join(in_deg,  "id", "left")
    .join(out_deg, "id", "left")
    .fillna(0, subset=["in_degree","out_degree"])
    .withColumn("total_degree", F.col("in_degree") + F.col("out_degree")))
degree_df.orderBy(F.desc("total_degree")).show(10, truncate=False)
degree_df.write.mode("overwrite").option("header", True).csv(f"{OUT}/q2b_degrees")
print("Q2b degrees saved")

# Q3 — BFS and shortest paths
clean_edges = edges.filter((F.col("trip_distance") > 0) & (F.col("total_amount") > 0)).cache()
g_clean = GraphFrame(vertices, clean_edges)
START, GOAL = 234, 132

try:
    bfs = g_clean.bfs(fromExpr=f"id = {START}", toExpr=f"id = {GOAL}", maxPathLength=4)
    print(f"BFS paths: {bfs.count()}")
    bfs.show(5, truncate=False)
    bfs.write.mode("overwrite").option("header", True).csv(f"{OUT}/q3_bfs")
except Exception as e:
    print(f"BFS: {e}")

try:
    sp = g_clean.shortestPaths(landmarks=[GOAL])
    sp_df = (sp.select("id","v_Zone","v_Borough","distances")
               .withColumn("hops", F.col("distances").getItem(GOAL))
               .drop("distances").filter(F.col("hops").isNotNull()).orderBy("hops"))
    sp_df.show(10, truncate=False)
    sp_df.write.mode("overwrite").option("header", True).csv(f"{OUT}/q3_shortest")
except Exception as e:
    print(f"shortestPaths: {e}")

# Q4a — classic PageRank
try:
    g_uniq = GraphFrame(vertices, clean_edges.select("src","dst").dropDuplicates())
    pr = g_uniq.pageRank(resetProbability=0.15, tol=0.01)
    top10_pr = pr.vertices.select("id","v_Zone","v_Borough","pagerank").orderBy(F.desc("pagerank")).limit(10)
    top10_pr.show(truncate=False)
    top10_pr.write.mode("overwrite").option("header", True).csv(f"{OUT}/q4a_pagerank")
    print("Q4a saved")
except Exception as e:
    print(f"PageRank classic: {e}")

# Q4b — frequency-weighted PageRank
try:
    freq_edges = clean_edges.groupBy("src","dst").agg(F.count("*").alias("weight"))
    pr_freq = GraphFrame(vertices, freq_edges).pageRank(resetProbability=0.15, tol=0.01)
    top10_freq = pr_freq.vertices.select("id","v_Zone","v_Borough","pagerank").orderBy(F.desc("pagerank")).limit(10)
    top10_freq.show(truncate=False)
    top10_freq.write.mode("overwrite").option("header", True).csv(f"{OUT}/q4b_pagerank_freq")
    print("Q4b saved")
except Exception as e:
    print(f"PageRank freq: {e}")

# Q5 — Label Propagation
try:
    lpa = g_clean.labelPropagation(maxIter=5)
    lpa_meta = (lpa.join(vertices.select("id","v_Borough","v_Zone"), "id")
                   .withColumnRenamed("label","community"))
    n_comm = lpa_meta.select("community").distinct().count()
    print(f"Communities: {n_comm}")
    top5 = lpa_meta.groupBy("community").agg(F.count("*").alias("size")).orderBy(F.desc("size")).limit(5)
    top5.show()
    largest = top5.first()["community"]
    lpa_meta.filter(F.col("community")==largest).groupBy("v_Borough").agg(F.count("*").alias("zones")).orderBy(F.desc("zones")).show()
    lpa_meta.write.mode("overwrite").option("header", True).csv(f"{OUT}/q5_communities")
    print("Q5 saved")
except Exception as e:
    print(f"LPA: {e}")

spark.stop()
print("Task 3 complete")
