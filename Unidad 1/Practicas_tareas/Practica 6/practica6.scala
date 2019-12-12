import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

import spark.implicits._

// 1
//scala notation
df.filter($"Close">480).show()

//sparksql notation
df.filter("Close > 480").show()

// 2
df.filter($"Close" < 480 && $"High" < 480).show()

df.filter("Close < 480 AND High < 480").show()
// 3
df.select(month(df("Date"))).show()

df.select(year(df($"Date"))).show()

val df2 = df.withColumn("Year", year(df("Date")))

// 4
val dfavgs = df2.groupBy($"Year").mean()

dfavgs.select($"Year", $"avg(Close)").show()

// 5
val dfmins = df2.groupBy($"Year").min()

dfmins.select($"Year", $"min(Close)").show()
