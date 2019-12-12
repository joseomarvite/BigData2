// Se importan las librerias necesarias para el trabajo de Kmeans
// Se crea un sesion en Spark
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
import org.apache.spark.ml.clustering.KMeans

// Se cargan los datos en la variable "dataset"
val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")

// Se entra el modelo de K means y a la vez se especifica el valor de K
val kmeans = new KMeans().setK(2).setSeed(1L)
// Se entrena el modelo con los datos de la variable "dataset"
val model = kmeans.fit(dataset)

// Se evalua el cluster calculando en los errores cuadraticos 
val WSSE = model.computeCost(dataset)
println(s"Within set sum of Squared Errors = $WSSE")

// Se muestran el movimiento de los centroides 
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
