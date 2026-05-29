from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("task1_expedia").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

DATA   = "s3a://module-big-data-processing/Expedia/hotels.csv"
OUT    = "s3a://bdp-student-ec251102/task1"

# - Q1
df = spark.read.option("header", True).option("inferSchema", True).csv(DATA)
df.cache()
df.printSchema()
print("Total impressions:", df.count())
print("Distinct searches:", df.select("srch_id").distinct().count())

# - Q2 
df = (df
    .withColumn("ts",    F.to_timestamp("date_time", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("date",  F.to_date("ts"))
    .withColumn("month", F.month("ts"))
    .withColumn("dow",   F.dayofweek("ts"))
    .withColumn("hour",  F.hour("ts"))
)
df.select("date_time", "date", "month", "dow", "hour").show(10, truncate=False)

# - Q3 
daily = (df.groupBy("date").agg(
    F.count("*")               .alias("impressions"),
    F.countDistinct("srch_id") .alias("searches"),
    F.sum("click_bool")        .alias("clicks"),
    F.sum("booking_bool")      .alias("bookings")
).withColumn("ctr",       F.round(F.col("clicks")   / F.col("impressions"), 4))
 .withColumn("conv_rate", F.round(F.col("bookings") / F.col("impressions"), 4))
 .orderBy("date"))

daily.show(10)
daily.write.mode("overwrite").option("header", True).csv(f"{OUT}/q3_daily")

# - Q4
tod = (df.groupBy("hour", "srch_saturday_night_bool").agg(
    F.count("*").alias("impressions"),
    F.round(F.sum("click_bool") / F.count("*"), 4).alias("ctr")
).orderBy("hour", "srch_saturday_night_bool"))

tod.show(24)
tod.write.mode("overwrite").option("header", True).csv(f"{OUT}/q4_tod")

# - Q5 
df = df.withColumn("star_band",
    F.when(F.col("prop_starrating") <= 2, "0-2")
     .when(F.col("prop_starrating") == 3, "3")
     .when(F.col("prop_starrating") == 4, "4")
     .otherwise("5"))

pos = (df.groupBy("position", "star_band").agg(
    F.sum("click_bool").alias("clicks"),
    F.round(F.sum("click_bool")   / F.count("*"), 4).alias("ctr"),
    F.round(F.sum("booking_bool") / F.count("*"), 4).alias("conv_rate")
).orderBy("position", "star_band"))

pos.show(20)
pos.write.mode("overwrite").option("header", True).csv(f"{OUT}/q5_position")

# - Q6 
dest = (df.groupBy("month", "srch_destination_id").agg(
    F.count("*")          .alias("impressions"),
    F.sum("booking_bool") .alias("bookings"),
    F.round(F.sum("booking_bool") / F.count("*"), 4).alias("conv_rate")
))

w_bk = Window.partitionBy("month").orderBy(F.desc("bookings"))
top_book = dest.withColumn("rk", F.rank().over(w_bk)).filter(F.col("rk") <= 10).drop("rk")
top_book.orderBy("month", F.desc("bookings")).show(20)
top_book.write.mode("overwrite").option("header", True).csv(f"{OUT}/q6_top_bookings")

# - Q7 
daily_s = df.groupBy("date").agg(F.countDistinct("srch_id").alias("searches")).orderBy("date")
mean_val = daily_s.agg(F.mean("searches")).collect()[0][0]
print(f"Mean daily searches: {mean_val:.1f}")

unusual = daily_s.filter(F.col("searches") > mean_val).orderBy(F.desc("searches"))
unusual.show()

daily_s.write.mode("overwrite").option("header", True).csv(f"{OUT}/q7_daily")
unusual.write.mode("overwrite").option("header", True).csv(f"{OUT}/q7_unusual")

spark.stop()
print("Task 1 complete — outputs at", OUT)
