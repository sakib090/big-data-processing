from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("task2_books").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

RATINGS = "s3a://module-big-data-processing/AmazonBooks/Books_rating.csv"
BOOKS   = "s3a://module-big-data-processing/AmazonBooks/books_data.csv"
OUT     = "s3a://bdp-student-ec251102/task2"

# - Q1 
ratings = (spark.read.option("header", True).option("inferSchema", True).csv(RATINGS)
           .filter(F.col("review_time").isNotNull())
           .filter(F.col("review_time").cast("long") > 0))

books = (spark.read.option("header", True).csv(BOOKS)
         .withColumn("categories",
             F.when(F.col("categories").isNull(), "Other").otherwise(F.col("categories"))))

ratings.printSchema()
books.printSchema()
print("Reviews:        ", ratings.count())
print("Distinct users: ", ratings.select("user_id").distinct().count())
print("Distinct books: ", ratings.select("id").distinct().count())

# - Q2
ratings = (ratings
    .withColumn("review_ts",   F.to_timestamp(F.col("review_time").cast("long")))
    .withColumn("review_date", F.date_format("review_ts", "yyyy-MM-dd"))
    .withColumn("year",        F.year("review_ts"))
    .withColumn("month",       F.month("review_ts"))
    .withColumn("half_year",
        F.when(F.month("review_ts").between(1, 6), "Early Year").otherwise("Late Year")))

ratings.orderBy("review_ts").select(
    "id", "review_score", "review_date", "year", "month", "half_year"
).show(10, truncate=False)

# - Q3 
ratings = ratings.withColumn("rating_band",
    F.when(F.col("review_score") <= 2, "Low")
     .when(F.col("review_score") <= 4, "Medium")
     .otherwise("High"))

total = ratings.count()
band_stats = (ratings.groupBy("rating_band").agg(
    F.count("*")              .alias("total_reviews"),
    F.countDistinct("user_id").alias("distinct_users"),
    F.countDistinct("id")     .alias("distinct_books"),
    F.round(F.count("*") / total * 100, 2).alias("pct_total")
))

per_user = ratings.groupBy("rating_band", "user_id").agg(F.count("*").alias("n"))
avg_pu   = per_user.groupBy("rating_band").agg(F.round(F.mean("n"), 2).alias("avg_per_user"))
band_stats = band_stats.join(avg_pu, "rating_band").orderBy(F.desc("total_reviews"))
band_stats.show()
band_stats.write.mode("overwrite").option("header", True).csv(f"{OUT}/q3_bands")

# - Q4 
time_df = (ratings.groupBy("year", "half_year").agg(
    F.count("*")                      .alias("total_reviews"),
    F.round(F.mean("review_score"), 3).alias("avg_score")
).orderBy("year", "half_year"))
time_df.show(10)

yearly = ratings.groupBy("year").agg(F.count("*").alias("reviews")).orderBy("year")
w = Window.orderBy("year")
yearly = (yearly
    .withColumn("prev",       F.lag("reviews").over(w))
    .withColumn("yoy_change", F.col("reviews") - F.col("prev"))
    .withColumn("yoy_pct",    F.round((F.col("reviews") - F.col("prev")) / F.col("prev") * 100, 2)))
yearly.show()

time_df.write.mode("overwrite").option("header", True).csv(f"{OUT}/q4_time")
yearly.write.mode("overwrite").option("header", True).csv(f"{OUT}/q4_yearly")

# - Q5 
joined = (ratings.join(books.select("id", "categories"), "id", "left")
          .withColumn("categories",
              F.when(F.col("categories").isNull(), "Other").otherwise(F.col("categories"))))

cat = (joined.groupBy("categories").agg(
    F.count("*")              .alias("total_reviews"),
    F.round(F.mean("review_score"), 3).alias("avg_score"),
    F.countDistinct("user_id").alias("distinct_users"),
    F.countDistinct("id")     .alias("distinct_books")
))
apb = (joined.groupBy("categories", "id").agg(F.count("*").alias("n"))
       .groupBy("categories").agg(F.round(F.mean("n"), 2).alias("avg_per_book")))

cat = cat.join(apb, "categories").orderBy(F.desc("total_reviews"))
cat.show(10, truncate=False)
cat.write.mode("overwrite").option("header", True).csv(f"{OUT}/q5_categories")

# - Q6 
book_joined = ratings.join(books.select("id", "title"), "id", "left")
top_books = (book_joined.groupBy("title").agg(
    F.count("*")              .alias("total_reviews"),
    F.round(F.mean("review_score"), 3).alias("avg_score"),
    F.countDistinct("user_id").alias("distinct_users")
).filter(F.col("total_reviews") >= 50)
 .orderBy(F.desc("avg_score"), F.desc("total_reviews"))
 .limit(10))

top_books.show(10, truncate=False)
top_books.write.mode("overwrite").option("header", True).csv(f"{OUT}/q6_top_books")

# - Q7 
user_counts = (ratings.groupBy("user_id").agg(F.count("*").alias("review_count"))
               .withColumn("reviewer_type",
                   F.when(F.col("review_count") > 50, "Frequent").otherwise("Infrequent")))

user_counts.orderBy(F.desc("review_count")).show(10, truncate=False)
user_counts.groupBy("reviewer_type").agg(F.count("*").alias("num_users")).show()
user_counts.write.mode("overwrite").option("header", True).csv(f"{OUT}/q7_users")

cat_rev = (joined.groupBy("categories")
           .agg(F.countDistinct("user_id").alias("distinct_reviewers"))
           .orderBy(F.desc("distinct_reviewers")).limit(5))
cat_rev.show(truncate=False)
cat_rev.write.mode("overwrite").option("header", True).csv(f"{OUT}/q7_cat_reviewers")

spark.stop()
print("Task 2 complete — outputs at", OUT)
