import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

// Se cargan los datos en la variable "data"
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

// Se imprime el esquema de los datos
data.printSchema()

// Se eliminan los campos null 
val dataClean = data.na.drop()

// Se declara un vector que transformara los datos a la variable "features"
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

// Se transforman los features usando el dataframe
val features = vectorFeatures.transform(dataClean)

// Se declara un "StringIndexer" que transformada los datos en "species" en datos numericos 
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

// Ajustamos las especies indexadas con el vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)

// Con la variable "splits" hacemos un corte de forma aleatoria
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
// Se declara la variable "train" la cual tendra el 60% de los datos
val train = splits(0)
// Se declara la variable "test" la cual tendra el 40% de los datos
val test = splits(1)

// Se establece la configuracion de las capas para el modelo de redes neuronales artificiales
// De entrada: 4
// Intermedias 2
// Salida 3, al tener 3 tipos de clases
val layers = Array[Int](4, 5, 4, 3)

// Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)

// Se prueban ya entrenado el modelo
val result = model.transform(test)

// Se selecciona la prediccion y la etiqueta que seran guardado en la variable 
val predictionAndLabels = result.select("prediction", "label")
// Se muestran algunos datos 
predictionAndLabels.show()

// Se ejecuta la estimacion de la precision del modelo
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictionAndLabels)
// Se imprime el error del modelo
println(s"Test Error = ${(1.0 - accuracy)}")


//D).Explique detalladamente la funcion matematica de entrenamiento que utilizo con sus propias palabras
/*
La funcion de entrenamiento es el conjunto de datos  que se divide en una parte utilizada para entrenar el modelo (60%)
y otra parte para las prueba (40%).
Esta funcion mediante un array hacemos las pruebas de entrenamiento mediante un ramdom
asi de esta manera se entrena el algoritmo y se costruye el modelo
*/

//E).Explique la funcion de error que utilizo para el resultado final
/*
 Esta funcion nos sirve para calcular el error de prueba nos ayuda a medir la precisión del modelo utilizando el evaluador.
 y asi imprimir con exactitud el error de nuestro problema.
*/
