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
