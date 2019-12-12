# Datos Masivos - Unidad 3

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 3
 - **Nombre**: Vite Hernández José Omar
 - **No. Control**: 15211706
 - **Docente**: Dr. Jose Christian Romero Hernandez

# Evaluación 
**Examen 1**

**Introducción**

En el presente documento se podrá observar el conocimiento y procedimiento que se realizó en la 1ra evaluación de la unidad 3, el cual fue programado en el lenguaje Scala y se utilizaron librerías de Machine Learning de Spark en 4.4

**Explicación**

KNN es un método de clasificación supervisada (Aprendizaje, estimación basada en un conjunto de entrenamiento y prototipos)

El algoritmo KNN asume que existen cosas similares en la proximidad. En otras palabras, cosas similares están cerca unas de otras, en otras palabras K-NN captura la idea de similitud (a veces llamada distancia, proximidad o cercanía).

Existen muchas fórmulas para calcular la distancia, y una podría ser preferible dependiendo del problema que se esté resolviendo. Sin embargo, la distancia en línea recta (también llamada distancia euclidiana) es una opción popular y familiar.

**Algoritmo de KNN**

1.  Cargar los datos.
2.  Inicializa K a tu número elegido de vecinos.
3.  Para cada ejemplo en los datos.
4.  Calcule la distancia entre el ejemplo de consulta y el ejemplo actual a partir de los datos.
5.  Agregue la distancia y el índice del ejemplo a una colección ordenada.
6.  Ordene la colección ordenada de distancias e índices de menor a mayor (en orden ascendente) por las distancias.
7.  Elija las primeras K entradas de la colección ordenada.
8.  Obtener las etiquetas de las entradas K seleccionadas.
9.  Si es regresión, devuelve la media de las etiquetas K.
10.  Si es clasificación, devuelve el modo de las etiquetas K
    
**Elegir el valor correcto para K**

Para seleccionar la K adecuada para sus datos, ejecutamos el algoritmo KNN varias veces con diferentes valores de K y seleccionamos la K que reduce la cantidad de errores que encontramos al tiempo que mantenemos la capacidad del algoritmo para hacer predicciones con precisión cuando se le dan datos que no tienen.

Aquí hay algunas cosas a tener en cuenta:
-   A medida que disminuimos el valor de K a 1, nuestras predicciones se vuelven menos estables.
   
-   Inversamente, a medida que aumentamos el valor de K, nuestras predicciones se vuelven más estables debido a la mayoría de votos / promedios, y por lo tanto, es más probable que hagan predicciones más precisas (hasta cierto punto). Con el tiempo, comenzamos a presenciar un número creciente de errores. Es en este punto que sabemos que hemos empujado el valor de K demasiado lejos.
    
-   En los casos en que estamos tomando un voto mayoritario (por ejemplo, seleccionando el modo en un problema de clasificación) entre las etiquetas, generalmente hacemos de K un número impar para tener un desempate.
    
 **Ventajas**

1.  El algoritmo es simple y fácil de implementar.
2.  No hay necesidad de construir un modelo, ajustar varios parámetros o hacer suposiciones adicionales.
3.  El algoritmo es versátil, puede usarse para clasificación, regresión y búsqueda.
    
**Desventajas**

1.  El algoritmo se vuelve significamente más lento a medida que aumenta el número de ejemplos y/o predictores / variables independientes.

**Código comentado** 

```scala
// 1. Se importa la sesion en Spark
// 4. Se importa la libreria de Kmeans para el algoritmo
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
// 7. Se importa Vector Assembler para el manejo de datos
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._
// 2. Se minimizan los errores
Logger.getLogger("org").setLevel(Level.ERROR)
// 3. Creamos la instancia de sesion en Spark
val spark = SparkSession.builder().getOrCreate()
// 5. Cargamos el dataset de Wholesale Customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale-customers-data.csv")
// 6. Seleccionamos las columnas que seran los datos "feature"
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
// 8. Se crea un nuevo objeto para las columnas de caractersiticas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
// 9. Se utiliza el objetivo "assembler" para transformar "feature_data"
val traning = assembler.transform(feature_data)
// 10. Se crea el modelo con K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(traning)
// Evaluamos el cluster calculando en los errores cuadraticos
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")
//Resultado
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```

# Practicas
**Practica 1**

**Introducción**

En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**

La regresión logística resulta útil para los casos en los que se desea predecir la presencia o ausencia de una característica o resultado según los valores de un conjunto de predictores. Es similar a un modelo de regresión lineal pero está adaptado para modelos en los que la variable dependiente es dicotómica. Los coeficientes de regresión logística pueden utilizarse para estimar la odds ratio de cada variable independiente del modelo. La regresión logística se puede aplicar a un rango más amplio de situaciones de investigación que el análisis discriminante.

  

Ejemplo. ¿Qué características del estilo de vida son factores de riesgo de enfermedad cardiovascular ? Dada una muestra de pacientes a los que se mide la situación de fumador, dieta, ejercicio, consumo de alcohol, y estado de enfermedad cardiovascular, se puede generar un modelo utilizando las cuatro variables de estilo de vida para predecir la presencia o ausencia de enfermedad cardiovascular en una muestra de pacientes. El modelo puede utilizarse posteriormente para derivar estimaciones de la odds ratio para cada uno de los factores y así indicarle, por ejemplo, cuánto más probable es que los fumadores desarrollen una enfermedad cardiovascular frente a los no fumadores.

  

Estadísticos. Casos totales, Casos seleccionados, Casos válidos. Para cada variable categórica: parámetro coding. Para cada paso: variables introducidas o eliminadas, historial de iteraciones, -2 log de la verosimilitud, bondad de ajuste, estadístico de bondad de ajuste de Hosmer-Lemeshow, chi-cuadrado del modelo ¡, chi-cuadrado de la mejora, tabla de clasificación, correlaciones entre las variables, gráfico de las probabilidades pronosticadas y los grupos observados, chi-cuadrado residual. Para las variables de la ecuación: coeficiente (B), error estándar de B, Estadístico de Wald, razón de las ventajas estimada (exp(B)), intervalo de confianza para exp(B), log de la verosimilitud si el término se ha eliminado del modelo. Para cada variable que no esté en la ecuación: estadístico de puntuación. Para cada caso: grupo observado, probabilidad pronosticada, grupo pronosticado, residuo, residuo estandarizado.

Métodos. Puede estimar modelos utilizando la entrada en bloque de las variables o cualquiera de los siguientes métodos por pasos: condicional hacia delante, LR hacia delante, Wald hacia delante, Condicional hacia atrás, LR hacia atrás o Wald hacia atrás.
Regresión logística: Consideraciones sobre los datos

Datos. La variable dependiente debe ser dicotómica. Las variables independientes pueden estar a nivel de intervalo o ser categóricas; si son categóricas, deben ser variables auxiliares o estar codificadas como indicadores (existe una opción en el procedimiento para codificar automáticamente las variables categóricas).

**Código comentado** 

```scala
//Importamos librerias necesarias con las que vamos a trabajar
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.types.DateType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
//Elimina varios avisos de warnings/errores inecesarios
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
//Creacion del dataframe para cargar el archivo csv
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
//Imprimimos el esquema del dataframe para visualizarlo
data.printSchema()
//Imprime la primera linea de datos del csv
data.head(1)
data.select("Clicked on Ad").show()
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
//Tomamos nuestros datos mas relevantes a una variables y tomamos clicked on ad como nuestra label
val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
val logregdataal = logregdataall.na.drop()
//Generamos nuestro vector de ensamble en un arrengo donde tomamos nuestros features
val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")
//Utilizamos la regresion lineal en nuestros datos con un 70% y 30% de datos.
val Array(training, test) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(assembler,lr))
//Creacion del modelo
val model = pipeline.fit(training)
//resultados de las pruebas con nuestro modelo
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
//Imprimimos nuestras metricas y la accuaricy de los calculos
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy
```

**Practica 2**

**Introducción** 

La siguiente práctica es el resultado de la exposición 2 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

  

**Explicación**

K-means es un algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en k grupos basándose en sus características. El agrupamiento se realiza minimizando la suma de distancias entre cada objeto y el centroide de su grupo o cluster. Se suele usar la distancia cuadrática.

  

El algoritmo consta de tres pasos:

  

Inicialización: una vez escogido el número de grupos, k, se establecen k centroides en el espacio de los datos, por ejemplo, escogiendo aleatoriamente.

Asignación objetos a los centroides: cada objeto de los datos es asignado a su centroide más cercano.

Actualización centroides: se actualiza la posición del centroide de cada grupo tomando como nuevo centroide la posición del promedio de los objetos pertenecientes a dicho grupo.

Se repiten los pasos 2 y 3 hasta que los centroides no se mueven, o se mueven por debajo de una distancia umbral en cada paso.

  

El algoritmo k-means resuelve un problema de optimización, siendo la función a optimizar (minimizar) la suma de las distancias cuadráticas de cada objeto al centroide de su cluster.

  

Los objetos se representan con vectores reales de d dimensiones (x1,x2,…,xn) y el algoritmo k-means construye k grupos donde se minimiza la suma de distancias de los objetos, dentro de cada grupo S={S1,S2,…,Sk} , a su centroide. El problema se puede formular de la siguiente forma:

  

minSE(μi)=minS∑i=1k∑xj∈Si∥xj−μi∥2(1)

  
  
  
  

donde S es el conjunto de datos cuyos elementos son los objetos xj representados por vectores, donde cada uno de sus elementos representa una característica o atributo. Tendremos k grupos o clusters con su correspondiente centroide μi .

  

En cada actualización de los centroides, desde el punto de vista matemático, imponemos la condición necesaria de extremo a la función E(μi) que, para la función cuadrática (1) es:

  

∂E∂μi=0⟹μ(t+1)i=1∣∣S(t)i∣∣∑xj∈S(t)ixj

  
  

y se toma el promedio de los elementos de cada grupo como nuevo centroide.

  

Las principales ventajas del método k-means son que es un método sencillo y rápido. Pero es necesario decidir el valor de k y el resultado final depende de la inicialización de los centroides. En principio no converge al mínimo global sino a un mínimo local.

  

Ejemplo de de movimiento de centroides:

  

![](https://lh6.googleusercontent.com/pfMiXoGqUH9dQ0DPZv0jqIGI-63MtaeEgNaST0q_OsdcRAYc00HsREMlM8OSj2seeOrX62azsL8vR3Wiv0zAeQfaYuJvM8gymjuO8MKVy9aeClCsZA71fTM3LvOR9FJfVR-1tsJ0)

**Código comentado**

```scala
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
```

```
