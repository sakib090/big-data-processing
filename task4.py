import os, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, IntegerType

spark = (SparkSession.builder
         .appName("task4_streaming")
         # ── KEY FIX: cluster has ANSI mode on by default which makes cast()
         #    throw on malformed rows instead of returning NULL.
         #    Disabling it lets bad rows (server header lines) silently become
         #    NULL and get filtered out, so real data rows come through fine.
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

CHKPT = "s3a://bdp-student-ec251102/task4/checkpoints"
host  = os.getenv("STREAMING_SERVER_SMOKE",  "smoke-detection.stream-emulator.svc.cluster.local")
port  = int(os.getenv("STREAMING_SERVER_SMOKE_PORT", "5551"))
print(f"Socket source: {host}:{port}")

# = Schema + Parser 
SCHEMA = StructType([
    StructField("UTC",          LongType(),    True),
    StructField("temperature",  DoubleType(),  True),
    StructField("humidity",     DoubleType(),  True),
    StructField("tvoc",         DoubleType(),  True),
    StructField("eco2",         DoubleType(),  True),
    StructField("raw_h2",       DoubleType(),  True),
    StructField("raw_ethanol",  DoubleType(),  True),
    StructField("pressure",     DoubleType(),  True),
    StructField("pm1",          DoubleType(),  True),
    StructField("pm25",         DoubleType(),  True),
    StructField("nc05",         DoubleType(),  True),
    StructField("nc1",          DoubleType(),  True),
    StructField("nc25",         DoubleType(),  True),
    StructField("cnt",          DoubleType(),  True),
    StructField("fire_alarm",   IntegerType(), True),
])

raw = (spark.readStream.format("socket")
       .option("host", host).option("port", port)
       .option("includeTimestamp", "true").load())

sp = F.split(F.col("value"), ",")
parsed = (raw.select(
    F.col("timestamp").alias("proc_time"),
    *[sp.getItem(i).cast(SCHEMA[i].dataType).alias(SCHEMA[i].name)
      for i in range(len(SCHEMA))]
)
.withColumn("event_time", F.to_timestamp(F.col("UTC").cast("long")))
# Drop rows where UTC is NULL — these are server header / status lines
.filter(F.col("UTC").isNotNull()))

parsed.printSchema()

# - Q1 - Raw stream sample
q1 = (parsed.writeStream.format("console")
      .option("truncate", False).option("numRows", 5)
      .outputMode("append").queryName("q1").start())
time.sleep(20)
q1.stop()

# - Q2 - Readings per 10-second event-time window 
wn_counts = (parsed
             .withWatermark("event_time", "5 seconds")
             .groupBy(F.window("event_time", "10 seconds"))
             .agg(F.count("*").alias("reading_count")))

q2 = (wn_counts.writeStream.format("console")
      .option("truncate", False).outputMode("append").queryName("q2").start())
time.sleep(35)
q2.stop()

# - Q3a - Low-humidity events
low_hum = (parsed
           .withWatermark("event_time", "10 seconds")
           .filter(F.col("humidity") < 50)
           .groupBy(F.window("event_time", "60 seconds", "30 seconds"))
           .agg(F.count("*").alias("low_humidity_count")))

q3a = (low_hum.writeStream.format("console")
       .option("truncate", False).outputMode("update").queryName("q3a").start())
time.sleep(35)
q3a.stop()

# - Q3b - Fire alarm event distribution
alarm_counts = (parsed
                .groupBy("fire_alarm")
                .agg(F.count("*").alias("alarm_count"))
                .orderBy(F.desc("alarm_count")))

q3b = (alarm_counts.writeStream.format("console")
       .option("truncate", False).outputMode("complete")
       .option("checkpointLocation", f"{CHKPT}/q3b").queryName("q3b").start())
time.sleep(25)
q3b.stop()

# - Q4a - Avg temperature and PM2.5 by fire alarm status 
alarm_agg = (parsed
             .groupBy("fire_alarm")
             .agg(F.round(F.avg("temperature"), 3).alias("avg_temp"),
                  F.round(F.avg("pm25"),        3).alias("avg_pm25")))

q4a = (alarm_agg.writeStream.format("console")
       .option("truncate", False).outputMode("complete").queryName("q4a").start())
time.sleep(30)
q4a.stop()

# - Q4b - High-risk readings 
risk = parsed.withColumn("high_risk",
    F.when((F.col("eco2") >= 415) & (F.col("humidity") >= 50), 1).otherwise(0))

q4b = (risk.groupBy("high_risk").agg(F.count("*").alias("risk_count"))
       .writeStream.format("console").option("truncate", False)
       .outputMode("complete").trigger(processingTime="15 seconds")
       .queryName("q4b").start())
time.sleep(40)
q4b.stop()

# - Q5 - TVOC alert vs high-risk confusion matrix 
labelled = (parsed
    .withColumn("high_risk",  F.when((F.col("eco2") >= 415) & (F.col("humidity") >= 50), 1).otherwise(0))
    .withColumn("tvoc_flag",  F.when(F.col("tvoc") >= 20, 1).otherwise(0))
    .withWatermark("event_time", "10 seconds"))

win_metrics = (labelled
    .groupBy(F.window("event_time", "60 seconds", "30 seconds"))
    .agg(F.sum("tvoc_flag").alias("tvoc_alert_count"),
         F.sum("high_risk").alias("risk_count"))
    .withColumn("window_tvoc_alert", F.when(F.col("tvoc_alert_count") >= 1, 1).otherwise(0))
    .withColumn("window_risk",       F.when(F.col("risk_count")        >= 1, 1).otherwise(0)))

confusion = (win_metrics
    .groupBy("window_tvoc_alert", "window_risk")
    .agg(F.count("*").alias("window_count"))
    .orderBy("window_tvoc_alert", "window_risk"))

q5 = (confusion.writeStream.format("console")
      .option("truncate", False).outputMode("complete").queryName("q5").start())
time.sleep(50)
q5.stop()

# - Q6 - Checkpoint fault-tolerance demo
CHKPT_Q6 = f"{CHKPT}/q6_recovery"
alarm_ft = (parsed.groupBy("fire_alarm")
            .agg(F.count("*").alias("alarm_count"))
            .orderBy(F.desc("alarm_count")))

print("Q6 Run 1 — writing checkpoint...")
run1 = (alarm_ft.writeStream.format("console").option("truncate", False)
        .outputMode("complete").option("checkpointLocation", CHKPT_Q6)
        .queryName("q6_run1").start())
time.sleep(25)
b1 = run1.lastProgress.get("batchId", "?") if run1.lastProgress else "?"
print(f"Stopping at batch {b1}")
run1.stop()

time.sleep(3)
print("Q6 Run 2 — recovering from checkpoint...")
run2 = (alarm_ft.writeStream.format("console").option("truncate", False)
        .outputMode("complete").option("checkpointLocation", CHKPT_Q6)
        .queryName("q6_run2").start())
time.sleep(25)
b2 = run2.lastProgress.get("batchId", "?") if run2.lastProgress else "?"
print(f"Resumed at batch {b2} (was {b1})")
run2.stop()

spark.stop()
print("Task 4 complete")