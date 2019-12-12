# Datos Masivos - Unidad 1

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 1
 - **Nombre**: Vite Hernández José Omar
 - **No. Control**: 15211706
 - **Docente**: Dr. Jose Christian Romero Hernandez

# Evaluación 
**Examen 1**

**Introducción**
En el presente documento se podrá observar el conocimiento y procedimiento que se realizó en la 1ra evaluación de la unidad 1, el cual fue programado en el lenguaje Scala

**Explicación**
Se agregó un arreglo el cual contenía de los diferentes puntajes a lo largo de un periodo.
Se utilizó una función el cual iba a contener todas las variables necesarias para el funcionamiento del programa, para recorrer el arreglo se hizo a través de la función (foreach) “para cada uno”, esta función nos ayudará a almacenar cada uno de los elementos del arreglo en la variable (score), y la variable (score) pasaría por dos condicionales, el cual determinará si es mayor o es menor, al entrar a una condicional la variable (puntajeAlto o puntajeMenor) aumentaría “n” veces, cada vez que esté entre o la condición se cumpla, el resultado del aumento será desplegado en la pantalla, dando a conocer al usuario cada vez que se rompe el record dado sus puntajes o resultados.

**Código comentado** 
```scala
val puntajes = Array(3,4,21,36,10,28,35,5,24,42)
// Se agrego un funcion que recibira un arreglo de numeros enteros (Int)
def breakingRecords(puntajes:Array[Int]):Unit=
{
// Se agrego el puntaje, que sera el contador que se ira moviento en los espacios del arreglo
var score = 0
// La funcion (.head) nos da la cabeza del arreglo
var max = puntajes.head
var min = puntajes.head
// Esta variable llevaran el conteo de puntaje alto
var puntajeAlto = 0
// Esta variable llevaran el conteo de puntaje bajo
var puntajeMenor = 0
// Se crea un (foreach) en el arreglo, que ira comparando la variable (score)
// en cada espacio del arreglo
puntajes.foreach(score => {
// Si el score es mayor a la cabeza del arreglo, entonces ese tomara el valor
// y se agregara al puntaje mas alto, el cual se ira sumando conforme
// pase todos los elementos del arreglo
if (score > max) {
max = score
println("Puntaje alto: " + score)
puntajeAlto = puntajeAlto + 1
}
// Si el score es menor a la cabeza del arreglo, entonces ese tomara el valor
// y se agregara al puntaje mas alto, el cual se ira sumando conforme
// pase todos los elementos del arreglo
if (score < min) {
min = score
println("Puntaje menor: " + score)
puntajeMenor = puntajeMenor + 1
}
})
// Las suma de los puntajes seran enviados a la pantalla
println(puntajeAlto + " " + puntajeMenor)
}
breakingRecords(puntajes)
```
**Examen 1.2**

**Introducción**
En el presente documento se podrá observar el conocimiento y procedimiento que se realizó en la 2da evaluación de la unidad 1, el cual fue programado en el lenguaje Scala utilizando un dataframe

**Explicación**
El examen consta de 11 reactivos, de los cuales se utilizó un dataframe llamado “Netflix_2011_2016.csv” de los cuales utilizaremos los datos almacenados en el archivo “datos separados por comas”

Estas son las explicaciones de los 11 reactivos:
1. Se importó la sesión Spark
2. Se cargó el archivo “Netflix_2011_2016.csv” en la variable “df”
3. Con la función “columns” podemos ver las columnas del dataframe
4. Con la función “printSchema” podemos ver la estructura del dataframe, por ejemplo que tipo de dato corresponden a “x” columna
5. Con la función “head(n)” podemos ver los primeros cinco datos de todas las columnas
6. Con la función “describe().show()” podemos aprender sobre el dataframe
7. Primero se declara otra variable (“df2”) la cual tendrá los mismos datos de la variable “df”, en dicha declaración se agregó otra columna que tendrá el nombre de “HV Ratio” y los datos serán el resultado de la división de los datos de la columna “High” y “Volume”
8. NO EXISTE DICHA COLUMNA EN EL DATAFRAME
9. La columna “Close” corresponde al último valor que se le dio en un periodo de tiempo
10. Con la función “select” podemos seleccionar los datos de cualquier columna, en ellas se agregó la función “max” y “min” para poder seleccionar solo el valor máximo de la columna “Volume” y el valor mínimo
11. a) Con la ayuda la función “filter” podemos filtrar dichos datos dependiendo de una condición previamente agregada, y la función “count” contará cuantos datos fueron filtrados
b) Se filtró y posteriormente se contaron los datos filtrados, estos mismos se multiplicaron con todos los datos existentes en la columna y el resultado fue multiplicado por 100, para saber su dicho porcentaje
c) Con la ayuda de la función “corr” podemos determinar la correlación de Pearson de dos columnas
d) Se agregó la columna "Year" del dato "Date", en la variable "yeardf" después se seleccionó a partir de la variable "yeardf" el año y el máximo de "High", los datos fueron agregados de mayor a menor y se dio como resultado el máximo de la columna "High" por año
e) Se agrego la columna "Month" del dato "Date", en la variable "monthdf" después se seleccionó a partir de la variable "monthdf" el mes y el promedio de "Close"

**Código Comentado** 
```scala
//1. Comienza una simple sesion Spark
import org.apache.spark.sql.SparkSession
//2. Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//3. Cuales son los nombre de las columnas
df.columns
//4. Como se llama el esquema
df.printSchema()
//5. Imprime las primeras 5 columnas
df.head(5)
//6. Usa describe () para aprender sobre el DataFrame
df.describe().show()
//7. Crea una nueva columna llamada "HV Ratio" que es la relacion entre el precio
// de la columna "High" frente a la columna "Volumen" de acciones negociadas por un dia
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
```

# Practicas
**Practica 1**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Instrucciones**
1.  Desarrollar un algoritmo en scala que calcule el radio de un circulo.
2.  Desarrollar un algoritmo en scala que me diga si un numero es primo.
3.  Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy ecribiendo un tweet".
4.  Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke".
5.  ¿Cual es la diferencia en value y una variable en scala?.
6.  Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416.

**Código comentado** 
```scala
//Practica 1

//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
print("Ingrese la circunferencia del CIRCULO: ")
// EL usuario ingresa el valor de la variable
val circunferencia: Double = scala.io.StdIn.readLine.toInt
// Se realizan los calculos necesarios
val radio: Double = circunferencia/(2*3.1416)
// Se manda a imprimir la variable radio
println("El radio del circulo es: " + radio)
//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
def esprimo2(i :Int) : Boolean = {
// Utilce datos booleanos para determinar si era o no primo
if (i <= 1)
false
else if (i == 2)
true
else
!(2 to (i-1)).exists(x => i % x == 0)
}
// Se ingresa el primer valor donde iniciara el limite inferior
print("Ingrese un numero: ")
val numero1: Int = scala.io.StdIn.readLine.toInt
// El segundo valor que sera el limite superior
print("Ingrese un numero: ")
val numero2: Int = scala.io.StdIn.readLine.toInt
// Se mandara a imprimir con el uso del (foreach) todos los numero que sea primos
// dada la lista agregada anteriormente
(numero1 to numero2).foreach(i => if (esprimo2(i)) println("%d es primo.".format(i)))
//3. Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy ecribiendo un tweet"
var bird = "tweet"
string conexion "Estoy escribiendo un Twwet"
println("Estoy escribiendo un %s",bird)
//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"
var variable = "Hola Luke soy tu padre!"
// La funcion (slice) nos permite impirmir datos de una variable, dando sus coordenadas de lo que queremos extraer
variable.slice(5,9)
//5. Cual es la diferencia en value y una variable en scala?
// Respuesta: Value(val) se le asigna un valor definido y no puede ser cambiado, en cambio una variable(var) puede
// ser cambiado en cualquier momento
//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416
var x = (2,4,5,1,2,3,3.1416,23)
println(x._7)
```
**Practica 2**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo.

**Instrucciones**
1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro".
2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla".
	Traer los elementos de "lista" "verde", "amarillo", "azul".
3.	Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5.
	Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a 																		conjuntos.
4.	Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20, 		   		"Luis", 24, "Ana", 23, "Susana", "27".
5.	Imprime todas la llaves del mapa.
6.	Agrega el siguiente valor al mapa("Miguel", 23).

**Código comentado**
```scala
// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
// Importamos la biblieria que nos permite hacer listas mutables
// ya que por defecto las listas son inmutables, esto quiere decir
// que no se puede modificar
import scala.collection.mutable.ListBuffer
var lista = collection.mutable.ListBuffer("rojo","blanco","negro")
// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
// Ya con las lista mutable se puedes agregar uno por uno los datos que queremos tener
lista += "verde"
lista += "amarillo"
lista += "azul"
lista += "naranja"
lista += "perla"
// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
// La funcion (slice) nos permite tomar los datos con el uso de sus coordenadas
// que se encuentran en la lista
lista slice (3,6)
// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
// La funcion (range) nos permite darle el rango que queremos que tenga nuestro arreglo
Array.range(1, 1000, 5)
// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
// La funcion (toSet) nos muestra datos no repetidos o duplicados
lista.toSet
// 6. Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
var mutmap = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Susana", 27))
// 6 a . Imprime todas la llaves del mapa
println(mutmap)
// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)
mutmap += ("Miguel" -> 23)
```
**Practica 3**

**Introducción**
En este presente documento se deriva de investigación de la sucesión de Fibonacci en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Instrucciones**
1. Programar los 5 algoritmos de la sucesión de Fibonacci

**Explicación**
Basicamente, la sucesion de Fibonacci se realiza sumando siempre los ultimos 2 numeros (Todos los numeros presentes en la sucesion se llaman numeros de Fibonacci) de los siguientes 5 algoritmos matematicos:

**Código comentado** 
```scala
// Primer Algoritmo  
// En este primer algoritmo se agregó una función que después de haber efectuado  
// las operaciones correspondientes la función nos dará un resultado (return)  
// este debe de ser un valor entero (Int)  
def  funcion(n: Int): Int =  
{  
// Si el numero ingresado en la función es menor a 2, el numero ingresado será regresado  
if (n<2)  
{  
return n  
}  
// En todo caso, la funcion hara una serie de sumas, y regresa el resultado  
else  
{  
return funcion(n-1) + funcion(n-2)  
}  
}  
funcion(9)  
// Segundo Algoritmo  
// En este segundo algortimo se agrego una funcion que despues de haber efectuado  
// las operaciones correspondientes la funcion nos dara un resultado (return)  
// este debe de ser un valor entero con puntos decimales (Double)  
def  funcion1(n: Double): Double =  
{  
//Si el numero ingresado en la funcion es menor a 2, el numero ingresado sera regresado  
if (n<2)  
{
**return n  
}  
// En todo caso, se hara lo siguiente  
else  
{  
// La formula matematica es mas extensa, pero yo decidi hacerla en partes paqueñas  
// para luego unirla en la variable (j)  
var p = ((1+(Math.sqrt(5)))/2)  
var a = Math.pow(p,n)  
var b = Math.pow((1-p),n)  
var c = Math.sqrt(5)  
var j = ((a-(b)))/(c)  
// EL resultado de (j) sera el resultado al regresar (return)  
return(j)  
}  
  
}  
funcion1(9)  
// Tercer Algoritmo  
// En este tercer algortimo se agrego una funcion que despues de haber efectuado  
// las operaciones correspondientes la funcion nos dara un resultado (return)  
// este debe de ser un valor entero (Int)  
def  funcion2(n: Int): Int =  
{  
var a = 0  
var b = 1  
var c = 0  
// Se inicia un ciclo (for) donde k = 1, empezara a ciclar hasta llegar a ser (n)  
// (n) representa el valor que se ingresara en la funcion  
for (k <- 1 to n)  
{  
// En funcion al ciclo (for) las variables (c,a,b) empezaran a cambiar**
su resultado  
// hasta haber terminado el ciclo (for)  
c = b + a  
a = b  
b = c  
}  
// El resultado sera retornado con (return)  
return(a)  
}  
funcion2(9)  
// Cuarto Algoritmo  
// En este cuarto algortimo se agrego una funcion que despues de haber efectuado  
// las operaciones correspondientes la funcion nos dara un resultado (return)  
// este debe de ser un valor entero (Int)  
def  funcion3(n: Int): Int =  
{  
var a = 0  
var b = 1  
// Se inicia un ciclo (for) donde k = 1, empezara a ciclar hasta llegar a ser (n)  
// (n) representa el valor que se ingresara en la funcion  
for(k <- 1 to n)  
{  
b = b + a  
a = b - a  
// En funcion al ciclo (for) las variables (b,a) empezaran a cambiar su resultado  
// hasta haber terminado el ciclo (for)  
}  
// El resultado sera retornado con (return)  
return(a)  
}  
funcion3(9)
// Quinto Algoritmo  
// En este quinto algoritmo se realiza una funcion que pide un valor entero (Int)  
// para luego retornar un valor entero con decimales(Double)  
def  funcion4(n: Int): Double =  
{  
// Se crea un arreglo que empieza de 0 hasta (n+1)  
val vector = Array.range(0,n+1)  
// Si la variable (n) es menor a 2, se regresa esa misma variable como resultado  
if (n < 2)  
{  
return (n)  
}  
// En caso contrario el vector con espacio (0) tendra un valor de cero(0)  
// y el vector con espacio (1) tendra valor de uno(1)  
else  
{  
vector(0) = 0  
vector(1) = 1  
// Se empieza a ciclar con un for el vector  
for (k <- 2 to n)  
{  
vector(k) = vector(k-1) + vector(k-2)  
}  
// El resultado sera la variable (n) en funcion al vector establecido  
return vector(n)  
}  
}  
funcion4(9)  
// Sexto Algoritmo  
// En este sexto algortimo se agrego una funcion que despues de haber efectuado  
// las operaciones correspondientes la funcion nos dara un resultado (return)  
// este debe de ser un valor entero con puntos decimales (Double)  
def  funcion5 (n: Double): Double =  
{  
// Si el valor ingresado es menor o igual a 0, entonces ese valor se vera regresado  
if (n<=0)  
{  
return (n)  
}  
// En caso contrario se tendra que hacer una serie se operaciones  
else  
{  
var i: Double = n - 1  
var auxOne: Double = 0  
var auxTwo: Double = 1  
var a: Double = auxTwo  
var b: Double = auxOne  
var c: Double = auxOne  
var d: Double = auxTwo  
// Empezara un ciclo (while) donde las variables empezaran a cambiar de valor  
// en funcion a la iteracion del ciclo  
while (i > 0)  
{  
// Si la variable (i) es impar, se haran operaciones diferentes  
if (i % 2 == 1)  
{  
auxOne = (d*b) + (c*a)  
auxTwo = ((d+(b*a)) + (c*b))  
a = auxOne  
b = auxTwo  
}  
// Si la variable (i) es par, se haran operaciones diferentes  
else  
{  
var pow1 = Math.pow(c,2)  
var pow2 = Math.pow(d,2)  
auxOne = pow1 + pow2  
auxTwo = (d*((2*(c)) + d))  
c = auxOne  
d = auxTwo  
}  
// La variable (i) empezará a cambiar de valor cada vez que se itere el ciclo  
// hasta que salga del ciclo, y se regrese la suma de (a+b)  
i = (i / 2)  
}  
return(a+b)  
}  
}
funcion5(9)
```
**Practica 4**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo.

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento en formato “csv”, se declararon diferentes funciones para la variable.

**Instrucciones**
1. Agregar 20 funciones básicas para el la variable "df".

**Código comentado** 
```scala
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.format("csv").option("header", "true").load("CitiGroup2006_2008")
// Agregar 20 funciones basica para el la variable "df"
//1
df.head()
//2
df.columns
//3
df.show()
//4
df.select("Date").show()
//5
df.printSchema()
//6
df.count
//7
df.describe()
//8
df.first()
//9
df.sort()
//10
df.filter($"Close" < 480 && $"High" < 480).show()
//11
df.select(corr("High", "Low")).show()
//12
df.select($"Close" < 500 && $"High" < 600).count()
//13
df.select(mean("High")).show()
//14
df.select(max("High")).show()
//15
df.select(min("High")).show()
//16
df.select(sum("High")).show()
//17
df.filter($"High"===484.40).show()
//18
df.filter($"High" > 480).count()
//19
df.select(month(df("Date"))).show()
//20
df.select(year(df("Date"))).show()
```
**Practica 5**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se utilizaron diferentes funciones más comunes para el uso de la variable “df”

**Instrucciones**
1. Agregar 5 funciones 

**Código comentado**
```scala
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
```
**Practica 6**

**Introducción**
En este presente documento se deriva de la explicación de diferentes clases en el Instituto Tecnológico de Tijuana de la materia Minería de Datos lo cual pude desarrollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

**Explicación**
Se declaro una variable “df” de la cual tomará los datos de un documento “csv” y se buscó utilizar la sintaxis de Scala y SparkSQL en la consulta de datos

**Instrucciones**
1. Agregar 5 funciones utilizando el sintaxis de Sparksql y Scala 

**Código comentado**
```scala
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

```

```
