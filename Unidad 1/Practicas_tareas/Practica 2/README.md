# Practica 2
**Tecnológico Nacional de México
Instituto Tecnológico de Tijuana
Subdirección académica
Departamento de Sistemas y Computación
Semestre: AGOSTO- DICIEMBRE 2019**
**Ingeniería en Tecnologías de la Información y Comunicaciones**
**Materia**: Datos Masivos
**Practica**: 2
**Nombre**: Vite Hernández José Omar
**No. Control**: 15211706
**Docente**: Jose Christian Romero Hernandez


# Introducción

En este presente archivo se deriva de la explicacion de diferentes clases en el Instituto Tecnologico de Tijuana de la materia Mineria de Datos
lo cual pude desarollar mis habilidades y aplicar mis conocimientos aprendidos durante el periodo

## Explicación 



## Codigo
```scala
//Practice 2
// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
import scala.collection.mutable.ListBuffer
var lista = collection.mutable.ListBuffer("rojo","blanco","negro")

// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
lista += "verde"
lista += "amarillo"
lista += "azul"
lista += "naranja"
lista += "perla"

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista slice (3,6)

// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
Array.range(1, 1000, 5)

// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
lista.toSet

// 6. Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
var mutmap = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Susana", 27))

// 6 a . Imprime todas la llaves del mapa
println(mutmap)

// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)
mutmap += ("Miguel" -> 23)'
```
## GITHUB

[Practica 1](https://github.com/joseomarvite/BigData/blob/Unidad_1/Practicas_tareas/Practica%201/practica1.scala)

```
