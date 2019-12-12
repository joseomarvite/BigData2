# Datos Masivos - Unidad 2

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 2
 - **Nombre**: Vite Hernández José Omar
 - **No. Control**: 15211706
 - **Docente**: Dr. Jose Christian Romero Hernandez

# Evaluación 
**Examen 1**

**Introducción**
En el presente documento se podrá observar el conocimiento y procedimiento que se realizó en la 1ra evaluación de la unidad 3, el cual fue programado en el lenguaje Scala y se utilizaron librerías de Machine Learning de Spark en 4.4

**Explicación**
Se desarrolló en el lenguaje Scala, las siguientes instrucciones utilizando las librerías de Machine Learning
Se cargo el dataset iris.csv y se realizó una limpieza de los datos necesaria para después ser procesado con el algoritmo Multilayer perceptron.
1. Se cargaron los datos en la variable "data"
2. Se imprimió el esquema de los datos
3. Se eliminaron los campos null
4. Se declarar un vector que transformara los datos a la variable "features"
5. Se transformaron los features usando el dataframe
6. Se declaró un "StringIndexer" que transformada los datos en "species" en datos numéricos
7. Se ajustó las especies indexadas con el vector features
8. Con la variable "splits" se hizo un corte de forma aleatoria
9. Se declaro la variable "train" la cual tendrá el 60% de los datos
10. Se declaro la variable "test" la cual tendrá el 40% de los datos
11. Se establece la configuración de las capas para el modelo de redes neuronales artificiales:
1.  Entrada: 4, al tener 4 variables en el dataset
2.  Intermedias 2
3.  Salida 3, al tener 3 tipos de clases
12. Se configuro el entrenador del algoritmo Multilayer con sus respectivos parámetros
13. Se entreno el modelo con los datos de entrenamiento
14. Se prueban los datos de prueba ya entrenado el modelo
15. Se seleccionó la predicción y la etiqueta que serán guardado en la variable
16. Se muestran algunos datos
17. Se ejecutó la estimación de la precisión del modelo
18. Se imprime el error del modelo
D) Explique detalladamente la función matemática de entrenamiento que utilizo con sus propias palabras
E) Explique la función de error que utilizó para el resultado final

**Código comentado** 
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
// Se cargan los datos en la variable "data"
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
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
La funcion de entrenamiento es el conjunto de datos que se divide en una parte utilizada para entrenar el modelo (60%)
y otra parte para las prueba (40%).
Esta funcion mediante un array hacemos las pruebas de entrenamiento mediante un ramdom
asi de esta manera se entrena el algoritmo y se costruye el modelo
*/
//E).Explique la funcion de error que utilizo para el resultado final
/*
Esta funcion nos sirve para calcular el error de prueba nos ayuda a medir la precisión del modelo utilizando el evaluador.
y asi imprimir con exactitud el error de nuestro problema.
*/
```

# Practicas
**Practica 1**

**Introducción**
La siguiente práctica es el resultado de la exposición 1 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**

**Correlation**
El coeficiente de correlación de Pearson es una prueba que mide la relación estadística entre dos variables continuas. Si la asociación entre los elementos no es lineal, entonces el coeficiente no se encuentra representado adecuadamente.

El coeficiente de correlación de Pearson puede tomar un rango de valores de +1 a -1. Un valor de 0 indica que no hay asociación entre las dos variables. Un valor mayor que 0 indica una asociación positiva. Es decir, a medida que aumenta el valor de una variable, también lo hace el valor de la otra. Un valor menor que 0 indica una asociación negativa; es decir, a medida que aumenta el valor de una variable, el valor de la otra disminuye.

La fórmula del coeficiente de correlación de Pearson es la siguiente:

![](https://lh5.googleusercontent.com/lT5LR_8yWobCOVQ14jjz3ftPPzkFlKKz5mkGnqrtbGhVT9mpGFA_xaA6LXgxEEG8TkQ3jJgwUeaNIDRS4fks1NYD3_ZklDJiR04EcY-0PFoyNvC52tDrl8QJ7_e4F8SSo0pLmidt)

Donde:

“x” es igual a la variable número uno, “y” pertenece a la variable número dos, “zx” es la desviación estándar de laCorrelation variable uno, “zy” es la desviación estándar de la variable dos y “N” es es número de datos.

**Hypothesis testing**  
Las estadísticas tienen que ver con los datos. Los datos por sí solos no son interesantes. Es la interpretación de los datoHypothesis testings lo que nos interesa. Utilizando las pruebas de hipótesis , tratamos de interpretar o sacar conclusiones sobre la población utilizando datos de muestra. Una prueba de hipótesis evalúa dos afirmaciones mutuamente excluyentes sobre una población para determinar qué afirmación es mejor respaldada por los datos de la muestra. Siempre que queramos hacer afirmaciones sobre la distribución de datos o si un conjunto de resultados es diferente de otro conjunto de resultados en el aprendizaje automático aplicado, debemos confiar en las pruebas de hipótesis estadísticas.

**Summarizer**

Proporciona estadística en resúmenes de vectores en el uso de grandes volúmenes de datos, las métricas son:

1.  Máximo
2.  Mínimo
3.  Promedio
4.  Varianza
5.  Número de no ceros 
6.  Así como resultados totales

**Código comentado** 
**Correlation**
```scala
// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
// Se declaran en la variable "df" los vectores donde tendran los datos a procesar
val data = Seq(
Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
Vectors.dense(4.0, 5.0, 0.0, 3.0),
Vectors.dense(6.0, 7.0, 0.0, 8.0),
Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)
// Se tranforma la variable "df" a dataframe donde los datos seran agregados a la columna "features"
val df = data.map(Tuple1.apply).toDF("features")
// Se declara la funcion de correlacion y se empieza a trabajar con ellos
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n  $coeff1")
// Se manda a imprimir la correlacion
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n  $coeff2")
```
**Hypothesis testing**
```scala
// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
// Se declaran en la variable "data" los vectores donde tendran los datos a procesar
val data = Seq(
(0.0, Vectors.dense(0.5, 10.0)),
(0.0, Vectors.dense(1.5, 20.0)),
(1.0, Vectors.dense(1.5, 30.0)),
(0.0, Vectors.dense(3.5, 30.0)),
(0.0, Vectors.dense(3.5, 40.0)),
(1.0, Vectors.dense(3.5, 40.0))
)
// Se tranforma la variable "data" a dataframe ("df"), este mismo tendra dos columnas llamadas "label", "features"
val df = data.toDF("label", "features")
// Se declara el modelo y sus se agregan sus respectivos parametros
val chi = ChiSquareTest.test(df, "features", "label").head
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")
```
**Summarizer**
```scala
// Se importan las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
// Se crear una variable llamada "data" la cual tendra los datos a procesar
val data = Seq(
(Vectors.dense(2.0, 3.0, 5.0), 1.0),
(Vectors.dense(4.0, 6.0, 7.0), 2.0)
)
// Se transforma la variable "data" en DataFrame con las columnas "features" , "weight"
val df = data.toDF("features", "weight")
df.show()
//Creamos una sumatoria utilizando los datos del DataFrame
val (meanVal, varianceVal) = df.select(Summarizer.metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
//Creamos una sumatoria
val (meanVal2, varianceVal2) = df.select(Summarizer.mean($"features"),Summarizer.variance($"features")).as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
```
**Practica 2**

**Introducción** *
La siguiente práctica es el resultado de la exposición 2 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Un árbol de decisión es una serie de nodos, un gráfico direccional que comienza en la base con un solo nodo y se extiende a los muchos nodos hoja que representan las categorías que el árbol puede clasificar. Otra forma de pensar en un árbol de decisión es como un diagrama de flujo, donde el flujo comienza en el nodo raíz y termina con una decisión tomada en las hojas.
Es una herramienta de apoyo a la decisión. Utiliza un gráfico en forma de árbol para mostrar las predicciones que resultan de una serie de divisiones basadas en características.
Aquí hay algunos términos útiles para describir un árbol de decisión:

  

![](https://lh5.googleusercontent.com/hRQ2NI-TF_qVmKmdlp6K-XpqWXb2IurIvjBM0KVJCA-1nJtguMfc_jOlj5oRCrhOIuANn6OMao7Lx2DdCjFzQqGTODVs8YtaqrRf3WjyyeLK7D8TmLts12NFThnR576zPX1MxPGS)

 - **Nodo raíz**: un nodo raíz está al comienzo de un árbol. Representa a toda la población que se analiza. Desde el nodo raíz, la población
   se divide de acuerdo con varias características, y esos subgrupos se
   dividen a su vez en cada nodo de decisión debajo del nodo raíz.
 - **División**: es un proceso de división de un nodo en dos o más subnodos.
 - **Nodo de decisión**: cuando un subnodo se divide en subnodos adicionales, es un nodo de decisión.
 - **Nodo hoja o nodo terminal**: los nodos que no se dividen se denominan nodos hoja o terminal.
 - **Poda**: la eliminación de los subnodos de un nodo primario se denomina poda. Un árbol se cultiva mediante la división y se contrae
   mediante la poda.
 - **Rama o subárbol**: una subsección del árbol de decisión se denomina rama o subárbol, del mismo modo que una parte de un gráfico se
   denomina subgrafo.
 - **Nodo padre y nodo hijo**: estos son términos relativos. Cualquier nodo que se encuentre debajo de otro nodo es un nodo secundario o
   subnodo, y cualquier nodo que preceda a esos nodos secundarios se
   llama nodo primario.

  

![](https://lh5.googleusercontent.com/dUsfp5izTGE2YwZLb5tn2H7pKXR0v03lk5U2qKa9dVGn8AMSXNIBtXqkxOEWMJ4owxOMZOHLDB9tdsiXavQ5pjk7QPdvtNx1BI_Fo_dLafgy6-Aue_HvVMpbIyhtFmqfFvFo_xFr)

**Código comentado**
```scala
import org.apache.spark.ml.Pipeline //Transformers: Convierte un DataFrame en otro agregando mas columnas, Estimators fit() que produce el modelo
import org.apache.spark.ml.classification.DecisionTreeClassificationModel // Todas las clases y methodos que se utilizaran en el modelo
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer} //Simétricamente a StringIndexer, IndexToString
//asigna una columna de índices de etiquetas a una columna que contiene las etiquetas originales como cadenas
// Se instancia los DataFreme en la variable "data" en el formato "libsvm"
// El archivo a cargar obligatoriamente debe estar estructurado al formato permite datos de entrenamiento escasos.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
// Se agrega otra columna de indices, donde se tomaron los datos de la columna "label" y se
// se transformaron a datos numericos, para poder manipularlos
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
//
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous.
// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos que seran evaluados posteriormente para ver la precisión del arbol,
// el reparto de estos datos seran de forma aleatoria
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
// Se declara el Clasificador de árbol de decisión y se le agrega la columna que sera las etiquetas (indices) y
// los valores que cada respectivo indice (caracteristicas)
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Convierte las etiquetas indexadas a las originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Crea el DT pipeline Agregando los index, label y el arbol juntos
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
// Se entrena el modelo con los datos del arreglo "trainingData" que es el 70% de los datos totales
val model = pipeline.fit(trainingData)
// Se hacen las predicciones al tomar los datos sobrantes que se llevo "testData" que es el 30%
val predictions = model.transform(testData)
// Se manda a imprimir la etiqueta, sus respectivos valores y la prediccion de la etiqueta
predictions.select("predictedLabel", "label", "features").show(5)
// Evalua el modelo y retorna la métrica escalar
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
// La variable "accuracy" tomara la acertación que hubo respecto a "predictedLabel" y "label"
val accuracy = evaluator.evaluate(predictions)
// Se manda a imprimir el resultado de error con respecto a la exactitud
println(s"Test Error = ${(1.0 - accuracy)}")
// Se guarda en la variable
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
// e imprime el arbol de decisiones
println(s"Learned classification tree model:\n  ${treeModel.toDebugString}")
```
**Practica 3**

**Introducción**
La siguiente práctica es el resultado de la exposición 3 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
El bosque aleatorio, como su nombre lo indica, consiste en una gran cantidad de árboles de decisión individuales que operan como un conjunto . Cada árbol individual en el bosque aleatorio escupe una predicción de clase y la clase con más votos se convierte en la predicción de nuestro modelo

![](https://lh3.googleusercontent.com/3I8p9wCq6-6-AxDzt1vifEmWEmsFXQFk7TLCXKfy2ugmPybc29mkgm_WIf0WsUR2GSTs3gFiptsUReBYO3DA9bFVjJ8vVu-PyqNKfvtcChQwnm8pyCYp9-E1WnFtYrGBMO0GBOUz)
El concepto fundamental detrás del bosque aleatorio es simple pero poderoso: la sabiduría de las multitudes. En ciencia de datos, la razón por la que el modelo de bosque aleatorio funciona tan bien es:

Una gran cantidad de modelos (árboles) relativamente no correlacionados que operan como comité superará a cualquiera de los modelos constituyentes individuales.

La baja correlación entre modelos es la clave. Al igual que las inversiones con bajas correlaciones (como acciones y bonos) se unen para formar una cartera que es mayor que la suma de sus partes, los modelos no correlacionados pueden producir predicciones de conjunto que son más precisas que cualquiera de las predicciones individuales. La razón de este maravilloso efecto es que los árboles se protegen entre sí de sus errores individuales (siempre que no se equivoquen constantemente en la misma dirección). Si bien algunos árboles pueden estar equivocados, muchos otros árboles estarán en lo correcto, por lo que, como grupo, los árboles pueden moverse en la dirección correcta.
Los requisitos previos para que un bosque aleatorio funcione bien son:

 - Es necesario que haya alguna señal real en nuestras funciones para
   que los modelos creados con esas funciones funcionen mejor que las
   conjeturas aleatorias.
 -   Las predicciones (y, por lo tanto, los errores) hechas por los árboles individuales deben tener bajas correlaciones entre sí.

**Código comentado** 
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
// Cargue y analice el archivo de datos, convirtiéndolo en un DataFrame.
val data = spark.read.format("libsvm").load("./sample_libsvm_data.txt")data.show()
// Índice de etiquetas, agregando metadatos a la columna de etiquetas.
// Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Identifica automáticamente las características categóricas y las indexa.
// Se establecen el maxCategories para que las entidades con > 4 valores distintos se traten como continuas.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
// Divide los datos en conjuntos de entrenamiento y prueba (30% para pruebas).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
// Entrena un modelo RandomForest.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
// Convierte las etiquetas indexadas de nuevo a etiquetas originales.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
// Indicadores de cadena y bosque en una tubería.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
// Modelo de tren. Esto también ejecuta los indexadores.
val model = pipeline.fit(trainingData)
// Hacer predicciones.
val predictions = model.transform(testData)
// Seleccione filas de ejemplo para mostrar.
predictions.select("predictedLabel", "label", "features").show(5)
// Seleccione (predicción, etiqueta verdadera) y calcule el error de prueba.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n  ${rfModel.toDebugString}")
```
**Practica 4**

**Introducción**
La siguiente práctica es el resultado de la exposición 4 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Supongamos que intentamos predecir el precio de una casa dada su edad, superficie y ubicación.

![](https://lh5.googleusercontent.com/xlxVr_LXmuz-9eEZkxkL-gHjL_Wcv6IJbwi-LDBBcOZRukERPUnhDuKe0s6jjtPkrLS4-Q9iC7R7fDtxgSX4zzEBPOYWiuCmIex21wt54SXamsxuJ0dlv_eqgs9SqW56FX_jF8y4)

Paso 1: Calcule el promedio de la etiqueta de destino

Al abordar los problemas de regresión, comenzamos con una hoja que es el valor promedio de la variable que queremos predecir. Esta hoja se utilizará como línea de base para abordar la solución correcta en los pasos siguientes.

![](https://lh6.googleusercontent.com/gVhMuhIxeYe-VyYie1oTF2mE2PAX3bkBOGj0RuXSPhLKV8cbjssvbSVeLr0V7S-Fi5PP7GCJABIqWoWn7dv5o3QEQ-7iTjb6pb8MOoaGA8XzguRcLTM3QTkrWDfo8PXZYotyVcl-)

  
  
  
  
  

Paso 2: calcular los residuos

  

Para cada muestra, calculamos el residuo con la fórmula anterior.

  

residual = valor real - valor predicho

  

En nuestro ejemplo, el valor predicho es igual a la media calculada en el paso anterior y el valor real se puede encontrar en la columna de precios de cada muestra. Después de calcular los residuos, obtenemos la siguiente tabla.

  

![](https://lh4.googleusercontent.com/Uy4oeXWtqVIKGNGwZdRU8ZGWZ8rUdGm5a5d5uppj6BABmizqof06TpZoGuN-hUFewSd9h0bkJiq_BLqVxeOKemJl3pnwgCdTIWYjou2jMbVVPIuWz2HNOWnasHrXlwZfltxjfK8G)

Paso 3: construya un árbol de decisión

  

A continuación, construimos un árbol con el objetivo de predecir los residuos. En otras palabras, cada hoja contendrá una predicción en cuanto al valor del residuo (no la etiqueta deseada).

  
  

![](https://lh5.googleusercontent.com/DQHx5x6J8LZoNK_Mj7DIeZjjeQEbNQe1gkV5ZxR5lCEDxU1F3AJVbYzoKzC4fuzxCAB2P3lfiu1MGNpMfHNPwbYKwHoizFaroxcm2VIRy4Pg4W1Rwj9H1B_PtsbyfLPDs5whFaU5)

Paso 4: predice la etiqueta de destino usando todos los árboles dentro del conjunto

  

Cada muestra pasa a través de los nodos de decisión del árbol recién formado hasta que alcanza un plomo dado. El residuo en dicha hoja se usa para predecir el precio de la vivienda.

  

![](https://lh3.googleusercontent.com/TPbqSqPRAxiBX9LsG5WGekzBD78z67rJL6M_d47cMPqBUXbeTvoEpEUnpFxLPMlNYGaGuHa-58RjerkF8IUoLVfJZdNb1CSiN8HdUwa2Sa82XF_LYIesLEa2K5ivgu4B9AKVwPqZ)

![](https://lh4.googleusercontent.com/kjsUYC4FGJAQuCuQ5gsEpPsGbrQavEN276fk6VrG-yhf0lywcHdM7IC0me3dU_WvaVDLlRzx2Dsoo8c63nvxJRaUlGyRFY8NRfuj36NdOYyEV-SJH4yIq2F5PEZC3tf0JcmpGyuX)


Paso 5: Calcule los nuevos residuos

Calculamos un nuevo conjunto de residuos restando los precios reales de la vivienda de las predicciones hechas en el paso anterior. Los residuos se utilizarán para las hojas del próximo árbol de decisión como se describe en el paso 3.

![](https://lh6.googleusercontent.com/GhMX3v1j8JYOxDRCjJnG970Hvm6Pl5IZbEy1mTWSBW7_-NKW0DT6THfNbUWdRKEZ2x39_xH9qRqR1pt6quH1AXYRIWc8GgZub9dVfrsh7ExD_9v1jSwnRXvJphxFZgBbgvlW_E61)

![](https://lh4.googleusercontent.com/onOfEPepmgyhfeHJK5Hx8AOKZQEmk4A8Wd4-qoOoWs8fcm-oxb6o8XvHJe40Ewx1QPu6g4R3Vb2UIcg-zEbYe1-XzkW0zeGWtfeliuRz1NYSeTFa_j_moNb4FLvzq8mo51uvfLSa)

Paso 6: repita los pasos 3 a 5 hasta que el número de iteraciones coincida con el número especificado por el hiperparámetro (es decir, el número de estimadores)

![](https://lh6.googleusercontent.com/nDoagpWXY27S_4Cm_fwPDX6N4L9tsEGZHxwtI1-Q5OmSQeumrS6ZNtUFWIaNAiMm2fm1w9X1qOcjTX8qIMGe7qlUizAXKYTeaAB4g5MX1nPx_OUIG7mZDZtAAGWxm6PdObVJ_p-f)

Paso 7: Una vez entrenado, use todos los árboles del conjunto para hacer una predicción final sobre el valor de la variable objetivo

La predicción final será igual a la media que calculamos en el primer paso, más todos los residuos pronosticados por los árboles que componen el bosque multiplicado por la tasa de aprendizaje.

  

![](https://lh4.googleusercontent.com/gurTQjlWyT9Zrn6dZS7qqaJrMnoUAqZ4LZxO8qZYAPBjJHAF_Z7AAtIwGWMQiBj37sfvVmGLN2X4zsXBa7rkSZ07FE3WcLFi6UaKPaXE8jIY78O54ar07m-MZU9wjVtZGLJIVRM1)

**Código comentado** 
```scala
// Se importan las librerias necesarias para trabajar con Clasificador de árbol impulsado por gradiente
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
// Se carga los datos en la variable "data" en formato "libsvm"
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
// Se agrega una nueva columna "IndexLabel" que tendra todos los datos de la columna "label" y tambien
// los datos los transforma en datos numericos para poder trabajar con ellos
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Se agrega una nueva columna "indexedFeatures" que tendra todos los datos de la columna "features" y tambien
// los datos los transforma en datos numericos para poder trabajar con ellos la diferencia en este al anterior
// es que se utiliza un vector ya que los "features" son mas de un datos
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
// Se declaran dos arreglos; "trainingData" y "testData" de los cuales tendran 70% y 30% de los datos
// que fueron declarados en la variable "data"
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
// Se declara el modelo y se agregan como parametros "indexedLabel" y "indexedFeatures", que son las etiquetas de cada clase
// y las caracteristicas de esa clase
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
// Se convierten las "indexedLabel" a las etiquetas originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
// Se declara el objeto "pipeline" en donde nos ayudara a pasar el codigo por estados, estos mismos estan declarados despues de "Array"
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
// Se entrena el modelo con los datos de entrenamiento
val model = pipeline.fit(trainingData)
// Se hacen las predicciones con el modelos ya entrenado y con los datos de prueba que representan el 30%
val predictions = model.transform(testData)
// Se mandan a imprimir o se seleccionan algunas columnas y se muestran solo las primerias 5
predictions.select("predictedLabel", "label", "features").show(5)
// Se evalua la precision y se agrega a una variable "accuracy"
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
// Se manda a imprimir el error de precision del modelo
println(s"Test Error = ${1.0 - accuracy}")
// Se manda a imprimir el arbol por medio de condicionales "if and else"
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n  ${gbtModel.toDebugString}")
```
**Practica 5**

**Introducción**
La siguiente práctica es el resultado de la exposición 5 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
El clasificador de perceptrón multicapa (MLPC) es un clasificador basado en la red neuronal artificial de alimentación directa. MLPC consta de múltiples capas de nodos. Cada capa está completamente conectada a la siguiente capa en la red. Los nodos en la capa de entrada representan los datos de entrada. Todos los demás nodos asignan entradas a salidas mediante una combinación lineal de las entradas con los pesos w y sesgo del nodo y aplicando una función de activación. Esto se puede escribir en forma de matriz para MLPC con capas K + 1 de la siguiente manera:

![](https://lh3.googleusercontent.com/xT9hKIYEV1XEsNKiRuWokD17F2F81YvAvrUetIglHDpVz_EkhPFHJxpZZX-D6r286BuKnUe78L81rUMslklhyZOIKtM8lzWVo9oJoYnTxyzkEMSV6WN4LbiEr0v9IK_WczbzL6rj)

Los nodos en las capas intermedias utilizan la función sigmoidea (logística):

![](https://lh6.googleusercontent.com/nqkyJMmGaiTgiV2BuKMfBHCqYJkzZuBra_OPk8798sPRjQhuqAM8F1ADa-cADNHxhEQSmFDhLsDGlAZgMqNC2Ep2SV6XdMKzMPa8MhjzjAw-9J1caQXF-dFD6-fWD426TsL4ZDtD)

Los nodos en la capa de salida utilizan la función softmax:

![](https://lh6.googleusercontent.com/l8tCf9Qp-Ig_Kx3_chkzcspuUHQtegs2nivDrxs8qxqd-HbixbxhjnG9rkoBgOdabEndM4SdCAoueYBU2uzpB7X51NlCuhRVkyZkWl0t8V7XPT3H4TDvrAPKUpKO0LZjo1cA0Ulm)

El número de nodos N en la capa de salida corresponde al número de clases.MLPC emplea la propagación hacia atrás para aprender el modelo. Utilizamos la función de pérdida logística para la optimización y L-BFGS como rutina de optimización.

**Código comentado**
```scala
// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Se cargan los datos en la variable "data" en un formato "libsvm"
val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
// Se declara un variable llamda "splits" donde se hacen los cortes de forma aleatoria de los datos de la variable "data"
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
// Se declara la variable "train" donde con ayuda de "splits" tendra el primer parametro que es el 60% de los datos cortados
val train = splits(0)
// Se declara la variable "test" donde con ayuda de "splits" tendra el primer parametro que es el 40% de los datos cortados
val test = splits(1)
// Se especifican las capas de la red neuronal
// Capa de entrada de tamaño 4 (características), dos intermedios de tamaño 5 y 4 y salida de tamaño 3 (clases)
val layers = Array[Int](4, 5, 4, 3)
// Se declara el modelo y se agregan los parametros necesarios para su funcionamiento
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)
// Se evalua y despliega resultados
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
// Se imprime el error de la precision
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
**Practica 6**

**Introducción**
La siguiente práctica es el resultado de la exposición 6 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Una máquina de vectores de soporte construye un hiperplano o un conjunto de hiperplanos en un espacio de dimensión alta o infinita, que puede usarse para clasificación, regresión u otras tareas. Intuitivamente, se logra una buena separación mediante el hiperplano que tiene la mayor distancia a los puntos de datos de entrenamiento más cercanos de cualquier clase (denominado margen funcional), ya que en general cuanto mayor es el margen, menor es el error de generalización del clasificador.

**Código comentado**
```scala
// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.LinearSVC
// Se cargan los datos en la variable "training" en un formato "libsvm"
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// Se entrena el modelo con todos los datos del archivo
val lsvcModel = lsvc.fit(training)
// Se manda a imprimir los coefcientes de super vector machine
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```
**Practica 7**

**Introducción**
La siguiente práctica es el resultado de la exposición 7 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
**OneVsRest** es un ejemplo de una reducción de aprendizaje automático para realizar una clasificación multiclase dado un clasificador base que puede realizar la clasificación binaria de manera eficiente. También se conoce como "Uno contra todos".

**OneVsRest** se implementa como un Estimador. Para el clasificador base, toma instancias de Clasificador y crea un problema de clasificación binaria para cada una de las k clases. El clasificador para la clase i está entrenado para predecir si la etiqueta es i o no, distinguiendo la clase i de todas las demás clases.

Las predicciones se realizan evaluando cada clasificador binario y el índice del clasificador más seguro se genera como etiqueta.

**Código comentado**
```scala
// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Secargan los datos en la variable "inputData" en un formato "libsvm"
val inputData = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos de prueba, respectivamente fueron declarados como arreglos y tendran el 80 y 20 porciento de los datos totales
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
// Se declara la variable "classifier" que hara la regresion
val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)
// Se declara el modelo "OneVsRest"
val ovr = new OneVsRest().setClassifier(classifier)
// Se entrena el modelo con los datos de entrenamiento
val ovrModel = ovr.fit(train)
// Se hacen las predicciones con los datos de prueba
val predictions = ovrModel.transform(test)
// Se declara el evaluador que tomara la precision del modelo y lo guardara en una variable metrica llamada "accuracy"
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
// Se calcula el error del modelo con una simple resta
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
```
**Practica 8**

**Introducción**
La siguiente práctica es el resultado de la exposición 8 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Los clasificadores Naive Bayes son una familia de clasificadores probabilísticos y multiclase simples basados ​​en la aplicación del teorema de Bayes con fuertes supuestos de independencia (Naive) entre cada par de características.

Naive Bayes puede ser entrenado de manera muy eficiente. Con un solo paso sobre los datos de entrenamiento, calcula la distribución de probabilidad condicional de cada característica dada cada etiqueta. Para la predicción, aplica el teorema de Bayes para calcular la distribución de probabilidad condicional de cada etiqueta dada una observación.

**Código comentado**
```scala
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Cargar los datos almacenados en formato LIBSVM como un DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
// Dividir los datos en conjuntos de entrenamiento y prueba (30% para pruebas)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
// Entrena un modelo NaiveBayes.
val model = new NaiveBayes().fit(trainingData)
// Seleccione filas de ejemplo para mostrar.
val predictions = model.transform(testData)
predictions.show()
// Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
```
```
