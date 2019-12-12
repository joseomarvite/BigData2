import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

//Clase 18 de septiembre 2019

//Agregar 5 funciones (Practica 5)

//1 .Funcion de una suma de una columna
df.select(sum("Volume")).show()

//2. Funcion del "minimo" de una columna
df.select(min("Volume")).show()

//3. Funcion del "maximo" de una columna
df.select(max("Volume")).show()

//4. Correlacion de dos columnas
df.stat.corr("High", "Low")

//5. Variancia en dos columnas
df.stat.cov("High", "Low")