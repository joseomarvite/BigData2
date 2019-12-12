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
