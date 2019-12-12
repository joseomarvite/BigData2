# Datos Masivos - Proyecto Final

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 4
 - **Nombre**: Vite Hernández José Omar
 - **No. Control**: 15211706
 - **Docente**: Dr. Jose Christian Romero Hernandez

# Evaluación 

**Introducción**
El presente documento es el resultado de una investigación sobre 3 diferentes tipos de aprendizaje automático en el rubro de la clasificación, estos algoritmos fueron desarrollados en el lenguaje de programación Scala con librerías (Machine Learning Library) de Spark en su versión 2.4.4

**Marco teórico de los algoritmos**

**Arbol de decision**

Un árbol de decisión es una serie de nodos, un gráfico direccional que comienza en la base con un solo nodo y se extiende a los muchos nodos hoja que representan las categorías que el árbol puede clasificar. Otra forma de pensar en un árbol de decisión es como un diagrama de flujo, donde el flujo comienza en el nodo raíz y termina con una decisión tomada en las hojas. Es una herramienta de apoyo a la decisión. Utiliza un gráfico en forma de árbol para mostrar las predicciones que resultan de una serie de divisiones basadas en características.

  

Aquí hay algunos términos útiles para describir un árbol de decisión:

  

![](https://lh4.googleusercontent.com/Qi4hDEjS4eraihFa8XTij4m0X7oCO6MlheW7prKg0H4hL7UNftGoSpvb_1nIVrqkIDVe29K1DkB1f9NM287aWovSH0SNb3O-rY4X3scWYIgdL30QlwpGLH59k3pPEB-16rQbknLC)

  

 - **Nodo raíz**: un nodo raíz está al comienzo de un árbol. Representa a toda la población que se analiza. Desde el nodo raíz, la población
   se divide de acuerdo con varias características, y esos subgrupos se
   dividen a su vez en cada nodo de decisión debajo del nodo raíz.

  

 - **División**: es un proceso de división de un nodo en dos o más subnodos.

  

 - **Nodo de decisión**: cuando un subnodo se divide en subnodos adicionales, es un nodo de decisión.

  

 - **Nodo hoja o nodo terminal:** los nodos que no se dividen se denominan nodos hoja o terminal.

  

 - **Poda**: la eliminación de los subnodos de un nodo primario se denomina poda. Un árbol se cultiva mediante la división y se contrae
   mediante la poda.

  

 - **Rama o subárbol**: una subsección del árbol de decisión se denomina rama o subárbol, del mismo modo que una parte de un gráfico se
   denomina subgrafo.

  

 - **Nodo padre y nodo hijo**: estos son términos relativos. Cualquier nodo que se encuentre debajo de otro nodo es un nodo secundario o
   subnodo, y cualquier nodo que preceda a esos nodos secundarios se
   llama nodo primario.

  

![](https://lh6.googleusercontent.com/pVMEyP0hFsBU2RS0NFCGj96Yab3mOyfoeI3164OE-Vt6bqOvSMY5u65p9basAyE5Ct2UzVWALtGvWfMfJbvD9rahlEZSk7Ru57hw1j9jv4X1CwrVhUppRa1WkZkSdesGtSuyGwgH)

 
**Regresión Logística**

La regresión logística resulta útil para los casos en los que se desea predecir la presencia o ausencia de una característica o resultado según los valores de un conjunto de predictores. Es similar a un modelo de regresión lineal pero está adaptado para modelos en los que la variable dependiente es dicotómica. Los coeficientes de regresión logística pueden utilizarse para estimar la odds ratio de cada variable independiente del modelo. La regresión logística se puede aplicar a un rango más amplio de situaciones de investigación que el análisis discriminante.

  

Ejemplo. ¿Qué características del estilo de vida son factores de riesgo de enfermedad cardiovascular ? Dada una muestra de pacientes a los que se mide la situación de fumador, dieta, ejercicio, consumo de alcohol, y estado de enfermedad cardiovascular, se puede generar un modelo utilizando las cuatro variables de estilo de vida para predecir la presencia o ausencia de enfermedad cardiovascular en una muestra de pacientes. El modelo puede utilizarse posteriormente para derivar estimaciones de la odds ratio para cada uno de los factores y así indicarle, por ejemplo, cuánto más probable es que los fumadores desarrollan una enfermedad cardiovascular frente a los no fumadores.

  

Estadísticos. Casos totales, Casos seleccionados, Casos válidos. Para cada variable categórica: parámetro coding. Para cada paso: variables introducidas o eliminadas, historial de iteraciones, -2 log de la verosimilitud, bondad de ajuste, estadístico de bondad de ajuste de Hosmer-Lemeshow, chi-cuadrado del modelo ¡, chi-cuadrado de la mejora, tabla de clasificación, correlaciones entre las variables, gráfico de las probabilidades pronosticadas y los grupos observados, chi-cuadrado residual. Para las variables de la ecuación: coeficiente (B), error estándar de B, Estadístico de Wald, razón de las ventajas estimada (exp(B)), intervalo de confianza para exp(B), log de la verosimilitud si el término se ha eliminado del modelo. Para cada variable que no esté en la ecuación: estadístico de puntuación. Para cada caso: grupo observado, probabilidad pronosticada, grupo pronosticado, residuo, residuo estandarizado.

  

Métodos. Puede estimar modelos utilizando la entrada en bloque de las variables o cualquiera de los siguientes métodos por pasos: condicional hacia delante, LR hacia delante, Wald hacia delante, Condicional hacia atrás, LR hacia atrás o Wald hacia atrás.

 
Regresión logística: Consideraciones sobre los datos

Datos. La variable dependiente debe ser dicotómica. Las variables independientes pueden estar a nivel de intervalo o ser categóricas; si son categóricas, deben ser variables auxiliares o estar codificadas como indicadores (existe una opción en el procedimiento para codificar automáticamente las variables categóricas).

  

**Perceptrón multicapa**

El clasificador de perceptrón multicapa (MLPC) es un clasificador basado en la red neuronal artificial de alimentación directa. MLPC consta de múltiples capas de nodos. Cada capa está completamente conectada a la siguiente capa en la red. Los nodos en la capa de entrada representan los datos de entrada. Todos los demás nodos asignan entradas a salidas mediante una combinación lineal de las entradas con los pesos w y sesgo del nodo y aplicando una función de activación. Esto se puede escribir en forma de matriz para MLPC con capas K + 1 de la siguiente manera:

![](https://lh6.googleusercontent.com/wTFvhkehzxs1qp-ukeHFlZyL3sTumjhLVO2mnvqjsn_SNwKgv3plIHfmBYsysIUevfu_lr09k-QFmxDr6CEIbFMtJsIPMSqqU1ChFDFB15nEjugXhcIzk0yPqk4T-azZQn6b2256)

Los nodos en las capas intermedias utilizan la función sigmoidea (logística):

![](https://lh4.googleusercontent.com/frmmxJP0n11G5fUxn3kMRpyb9oFiavIBWr-e62cV2Kc59RdrXv68UXtiGRCM1welJ7zPPO0N0osplaz2CyUGCsDSFwuvcLdLcDReNJtCeyDO_NGdSFSEyalvLDLhBHYswXbkAEQb)

Los nodos en la capa de salida utilizan la función softmax:

![](https://lh3.googleusercontent.com/fJhoJMWFuWkrizJP0baXI5gDyv1fxBgwjPtHfc3gxEqR6ldRwkHm1--9Gih9QvTirkGLgoT1LrNzjl45iBPpJALVRNtDD1sr0t0L1aSZfuLiT9TUmLVQ4ISkXh7-KNLIsWPnW9Qx)

El número de nodos N en la capa de salida corresponde al número de clases.MLPC emplea la propagación hacia atrás para aprender el modelo. Utilizamos la función de pérdida logística para la optimización y L-BFGS como rutina de optimización.

**Implementación**


Para su implementación se utilizó el lenguaje Scala con la librerías de Spark en su versión 2.4.4 y se utilizaron los datos del dataset “bank-full” en formato “.csv”, los cuales pueden ser consultados en la siguiente dirección:

[https://github.com/joseomarvite/BigData/tree/Unidad_4/Evaluacion](https://github.com/joseomarvite/BigData/tree/Unidad_4/Evaluacion)

  

Spark, permite trabajar con Scala, que es una forma abreviada de SCalable LAnguage, se originó en 'École Polytechnique Fédérale de Lausanne' (EPFL), Suiza, en 2003, con el objetivo de lograr un lenguaje de alto rendimiento y altamente concurrente que combine la fuerza de los siguientes dos patrones de programación líderes en la plataforma Java Virtual Machine (JVM): [6]

  

• Programación orientada a objetos

• programación funcional

  

API muy diferentes, pero algoritmos similares. Estas bibliotecas de aprendizaje automático heredan muchas de las consideraciones de rendimiento de las API RDD y Dataset en las que se basan, pero también tienen sus propias consideraciones. MLlib es la primera de las dos bibliotecas y está entrando en un modo de solo mantenimiento / corrección de errores.

  

Conjunto de datos

  

Los datos están relacionados con campañas de marketing directo de una institución bancaria portuguesa. Las campañas de marketing se basaron en llamadas telefónicas. A menudo, se requería más de un contacto con el mismo cliente para acceder si el producto (deposito bancario a plazo) estaría ('sí') o no ('no') suscrito.

  
  
  
  
  
  
  
  
  
  

Tomamos todos los ejemplos (41188) y 11 entradas, ordenadas por fecha (de Mayo de 2008 a noviembre de 2010).

De los datos a trabajar, usamos dos categorías de la información proveída y tomamos los siguientes:

  

1.  Información bancaria del cliente
    
2.  age números
    
3.  job: valores categóricos convertidos a números
    
4.  marital: estado civil categórico convertido a números
    
5.  education: nivel de estudios categóricos convertido a números
    
6.  default: si cuenta con algún tipo de crédito, datos categóricos convertidos a números
    
7.  housing: si renta casa, numérico
    
8.  loan: si tiene algún crédito personal, numérico
    

  

Información del cliente relacionada a la campaña actual y el último contacto

  

campaign: cuantas veces se ha contactado al cliente para la campaña actual

days: cuántos días han pasado desde el último contacto de la campaña

previous: cuantas veces se ha contactado al cliente anteriormente

outcome: cómo han sido los contactos anteriores.

  

Decision Tree

  

Implementar el algoritmo Decision Tree en Spark con Scala primero necesitamos importar las librerías o paquetes que contienen las funciones necesarias para utilizar este algoritmo.

El algoritmo necesita las características que se van a procesar para poder generar una predicción así que generamos un VectorAssembler que su funciones tomar conjuntos de datos y crear un vector que contenga los datos que a su vez es agregado a una nueva columna llamada features. Al final se transforma el vector generado con el dataframe creado.

  

Este algoritmo necesita tener indexados los datos antes de su procesamiento así que se genera un StringIndexer para transformar los label de tipo string a numérico y se ajustan a los datos indexados de igual forma se crea un vector indexer.

  

Se declara un arreglo, que contiene el 70% de los datos que será utilizado para entrenamiento y el 30% restante se utilizará para datos de pruebas. Después se declara la función de Decision Tree con las columnas label y features.

  

Se indexa la columna prediction con las etiquetas indexadas. Se crea una pipeline con las labels indexadas ,features indexadas, la instancia del algoritmo y la prediction indexada. Se genera un modelo con los datos de entrenamiento procesados.

  

Una vez que el modelo esté procesado con los datos de entrenamiento se generan las predicciones con los datos de prueba. Se imprimen 5 predicciones con sus label y features.

La variable evaluator guarda la precisión de las predicciones creadas y después se imprime la precisión que tuvo este algoritmo y muestra el error obtenido.

La variable treeModel muestra de forma gráfica los árboles de decisión creados a partir de los datos de prueba.

  

Multilayer Perceptron Classifier

Implementar el algoritmo Multilayer Perceptron Classifier en Spark con Scala primero necesitamos importar las librerías o paquetes que contienen las funciones necesarias para utilizar este algoritmo.

  

La primera línea importa un reductor de errores y la tercera línea declara una Sesión de Spark.

  
  

El algoritmo necesita las características que se van a procesar para poder generar una predicción así que generamos un VectorAssembler que su funciones tomar conjuntos de datos y crear un vector que contenga los datos que a su vez es agregado a una nueva columna llamada features. Al final se transforma el vector generado con el dataframe creado. Se renombra la columna Y con el nombre label, se genera un nuevo conjunto de datos seleccionado las columnas label y features.

  

Se declara un arreglo que contiene el 70% de los datos que será utilizado para entrenamiento y el 30% restante se utilizará para datos de pruebas tomados de los datos indexados aleatoriamente.

Se genera la configuración de las layers del algoritmo y se inicializa el trainer del MLPC agregando la configuración de las layers y el máximo de iteraciones.

  

Se procesa el modelo con los datos de entrenamiento y después se guardan los resultados utilizando los datos de pruebas que a su vez generan las predicciones. Se inicializa el evaluator que estima la precisión que se tuvo con este algoritmo utilizando el data frame creado con los datos indexados.

  

Logistic Regression

  

Implementar el algoritmo Logistic Regression en Spark con Scala primero necesitamos importar las librerías o paquetes que contienen las funciones necesarias para utilizar este algoritmo.

El algoritmo necesita las características que se van a procesar para poder generar una predicción así que generamos un VectorAssembler que su funciones tomar conjuntos de datos y crear un vector que contenga los datos que a su vez es agregado a una nueva columna llamada features. Al final se transforma el vector generado con el dataframe creado. Se renombra la columna Y con el nombre label, se genera un nuevo conjunto de datos seleccionado las columnas label y features.

  

Se declara un arreglo que contiene el 70% de los datos que será utilizado para entrenamiento y el 30% restante se utilizará para datos de pruebas tomados de los datos indexados aleatoriamente.

  

Se genera inicializa el modelo de Logistic Regression con el máximo de iteraciones.

Se procesa el modelo con los datos de entrenamiento y después se guardan los resultados utilizando los datos de pruebas que a su vez generan las predicciones. Se inicializa el evaluator que estima la precisión que se tuvo con este algoritmo utilizando el data frame creado con los datos indexados.

Se imprime la matriz de confusión generada a partir de las predicciones.
**Códigos**

**Arbol de decision**
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)
//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//DecisionTree
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
// features con mas de 4 valores distinctivos son tomados como continuos
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
//Division de los datos entre 70% y 30% en un arreglo
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
//Creamos un objeto DecisionTree
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Rama de prediccion
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Juntamos los datos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//Create a model of the entraining
val model = pipeline.fit(trainingData)
//Transformacion de datos en el modelo
val predictions = model.transform(testData)
//Desplegamos predicciones
predictions.select("predictedLabel", "label", "features").show(5)
//Evaluamos la exactitud
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n  ${treeModel.toDebugString}")
```

**Regresión Logística**
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)
//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Logistic Regresion
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
```
**Perceptrón multicapa**
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._
//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)
//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)
//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show(1)
//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Multilayer perceptron
//Dividimos los datos en un arreglo en partes de 70% y 30%
val split = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1)
// Especificamos las capas para la red neuronal
//De entrada 5 por el numero de datos de las features
//2 capas ocultas de dos neuronas
//La salida de 4 asi lo marca las clases
val layers = Array[Int](5, 2, 2, 4)
//Creamos el entrenador con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Entrenamos el modelo
val model = trainer.fit(train)
//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```


**Resultados**

**![](https://lh6.googleusercontent.com/J0lY9DUKc9rfTtpDDgh9T0jMwAlPQhOu1MQ7eX5Nm5ooLIlp1m-eRr3czfVUW-iIIRK8vGjAm1PIATgxgJ4ut24FGuk8t4YPaPe2a2KhaZS-iE7T5HpHF8sz2xiImEFOdsmLk43X)**
**![](https://lh5.googleusercontent.com/FKKvKDbEDVbp-BAHTzoBiElsC1DX3HZg-VMrw9AiR8sZJ1s1peMGeGAw6Of7oaIzt31a_SU8eSXr7iZX_Dp-p_oEsS1Snv6uNHaoFGaxHCI0RYESyFKCtzOukwJU0edN-2L-iW-P)**

**Conclusión**

Podemos concluir en base a los resultados obtenidos que al algoritmo de “Árboles de decisión” es más efectivo debido a la exactitud obtenida, también cabe destacar que los datos deben de tener sentido para que el algoritmo pueda funcionar correctamente.

Es importante recordar que los datos son tomados de la siguiente forma:
```scala

val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
```

“randomSplit” nos permite partir los datos de forma aleatoria, esto quiere decir que si compilamos el algoritmo más de una vez, existe la posibilidad que la exactitud aumente o disminuya ya que los datos fueron tomados de forma aleatoria, hay una gran posibilidad que en el 70% de los datos que se van a la variable “trainingData” todos tengan sentido al momentos de procesarlos, pero si en la variable “”testData” (30%) es todo lo contrario la exactitud disminuye de forma notable.

El resultado de “Árboles de decisión” o el despliegue del árbol está formulado en las condicionales “if and else” que nos permite graficar de forma sencilla mostrando la predicción en cada hoja terminal, esta sencillez hace que cualquier persona con nociones basicas de matematicas pueda entenderlo y explicarlo.
Los árboles de decisión son de gran herramienta para instituciones medicinales,
bancarios, administrativos, y sobre todo para la toma de decisiones.
Por ejemplo, en Colombia estos algoritmos tienen gran relevancia en bancos, donde hacen un análisis previo para determinar el riesgo crediticio cuando una persona física o moral hace uso de un préstamo mostrando probabilidades (Naive Bayes), ventajas y desventajas.

