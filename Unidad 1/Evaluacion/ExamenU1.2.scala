//1. Comienza una simple sesion Spark

import org.apache.spark.sql.SparkSession


//2. Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3.  Cuales son los nombre de las columnas

df.columns

//4.  Como se llama el esquema

df.printSchema()

//5.  Imprime las primeras 5 columnas

df.head(5)

//6.  Usa describe () para aprender sobre el DataFrame

df.describe().show()

//7.  Crea una nueva columna llamada "HV Ratio" que es la relacion entre el precio
//    de la columna "High" frente a la columna "Volumen" de acciones negociadas por un dia

// Se agrego la nueva columna "HV Ratio" la cual tendra el resultado de la division de "High" entre "Volume"

val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.show()

//9. ¿Cual es el significado de la columna Cerrar "Close"?
// Respuesta: Es el cierre del mes

//10. ¿Cual es el maximo y minimo de la columna "Volumen"?
df.select(max("Volume")).show()
df.select(min("Volume")).show()

//11. Con sintaxis Scala/Spark $ conteste los siguientes: 

//a) ¿Cuantos dias fue la columna "Close" inferior a $600?
df.filter($"Close"<600).count()
 

// b) ¿Que porcentaje del tiempo fue la columna "High" mayor que $500?
(df.filter($"High" > 500).count() * 1.0/ df.count())*100

// c) ¿Cual es la correlacion de Pearson entre Columna "High" y la columna "Volume"?
df.select(corr("High","Volume")).show()

// Se importa la sintaxis de SPARK para la consulta de datos 
import spark.implicits._

// d) ¿Cual es el maximo de la columna "High" por año?
// Se agrego la columna "Year" del dato "Date", en la variable "yeardf"
val yeardf = df.withColumn("Year",year(df("Date")))
// Se selecciono a partir de la variable "yeardf" el año y el maximo de "High", a partir de maximos a minimo en los años
val yearmaxs = yeardf.select($"Year",$"High").groupBy("Year").max()
// Se dio como resultado el maximo de la columna "High" por año
val res = yearmaxs.select($"Year",$"max(High)")
res.orderBy("Year").show()

// e) ¿Cual es el promedio de la columna "Close" para cada mes del calendario?
// Se agrego la columna "Month" del dato "Date", en la variable "monthdf"
val monthdf = df.withColumn("Month",month(df("Date")))
// Se selecciono a partir de la variable "monthdf" el mes y el promedio de "Close"
val monthavgs = monthdf.select($"Month",$"Close").groupBy("Month").mean()
// Se selecciono y mostro el promedio de la columna "Close" para cada mes del calendario
monthavgs.select($"Month",$"avg(Close)").orderBy("Month").show()

