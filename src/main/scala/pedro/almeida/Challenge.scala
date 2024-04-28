package pedro.almeida

import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{avg, col, collect_set, count, desc, explode, lit, max, split, to_date, udf, when}

object Challenge {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master(master = "local").appName(name = "XpandIT-Challenge").getOrCreate()

    val googlePlayStoreApps = spark.read.option("header", value = true).csv("src/main/resources/googleplaystore.csv")
    val googlePlayStoreReviews = spark.read.option("header", value = true).csv("src/main/resources/googleplaystore_user_reviews.csv")

    println("######## Part 1 ########")

    val df_1 = googlePlayStoreReviews
      .where(!col("Sentiment_Polarity").isNaN)
      .groupBy("App")
      .agg(avg("Sentiment_Polarity").cast("double").as("Average_Sentiment_Polarity"))

    df_1.printSchema()
    df_1.show(20, true)


    println("######## Part 2 ########")

    val df_2 = googlePlayStoreApps
      .where(col("Rating") >= 4.0 && !col("Rating").isNaN)
      .orderBy(desc("Rating"))

    df_2.write.option("header", value = true).option("delimiter", "§").mode(SaveMode.Overwrite).csv("src/main/resources/best_apps.csv")

    df_2.printSchema()
    df_2.show(20, true)


    println("######## Part 3 ########")

    val convertSizeToMB = udf((size: String) => {
      if (size.endsWith("M")) {
        size.dropRight(1)
      } else if (size.endsWith("k")) {
        (size.dropRight(1).toDouble / 1000).toString
      } else {
        size
      }
    })

    val convertDollarsToEuros = udf((input: String) => {
      if (input.startsWith("$")) {
        (input.drop(1).toDouble * 0.9).toString
      } else {
        input
      }
    })

    val df_3 = googlePlayStoreApps
      .withColumn("Rating", when(col("Rating").isNaN, lit(null)).otherwise(col("Rating")))
      .withColumn("Size", convertSizeToMB(col("Size")))
      .withColumn("Price", convertDollarsToEuros(col("Price")))
      .withColumn("Genres", split(col("Genres"), ";"))
      .withColumn("Last Updated", to_date(col("Last Updated"), "MMMM d, yyyy"))
      .groupBy("App")
      .agg(collect_set("Category").as("Categories"),
        max("Rating").cast("Double").as("Rating"),
        max("Reviews").cast("Long").as("Reviews"),
        max("Size").cast("Double").as("Size"),
        max("Installs").as("Installs"),
        max("Type").as("Type"),
        max("Price").cast("Double").as("Price"),
        max("Content Rating").as("Content_Rating"),
        max("Genres").as("Genres"),
        max("Last Updated").as("Last_Updated"),
        max("Current Ver").as("Current_Version"),
        max("Android Ver").as("Minimum_Android_Version")
      )

    df_3.printSchema()
    df_3.show(20, true)


    println("######## Part 4 ########")

    val df_4 = df_3
      .join(df_1, df_1.col("App") === df_3.col("App"), "left")
      .drop(df_1.col("App"))

    df_4.write.option("header", value = true).option("delimiter", "§").option("compression", "gzip").mode(SaveMode.Overwrite).parquet("src/main/resources/googleplaystore_cleaned")

    df_4.printSchema()
    df_4.show(20, true)


    println("######## Part 5 ########")

    val df_5 = df_4
      .withColumn("Genre", explode(col("Genres")))
      .groupBy("Genre")
      .agg(count("*").as("Count"),
        avg("Rating").as("Average_Rating"),
        avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity")
      )

    df_5.write.option("header", value = true).option("delimiter", "§").option("compression", "gzip").mode(SaveMode.Overwrite).parquet("src/main/resources/googleplaystore_metrics")

    df_5.printSchema()
    df_5.show(20, true)


    spark.stop()

  }
}