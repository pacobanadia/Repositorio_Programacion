# Programa que realiza operaciones matemáticas trigonométricas

import math


nombre = input("Ingrese su nombre: ")
apellido = input("Ingrese su apellido: ")
print("Hola " + nombre.strip().lower().capitalize() + " " + apellido.strip().lower().capitalize() + " Bienvenido a la calculadora de operaciones matematicas")

def calculadora():
    while True:
        print("CALCULADORA CON FUNCIONES TRIGONOMÉTRICAS")
        print("OPERACIONES DISPONIBLES:")
        print("1. Seno")
        print("2. Coseno")
        print("3. Tangente")
        print("4. Cotangente")
        print("5. Secante")
        print("6. Cosecante")
        print("7. Suma")
        print("8. Resta")
        print("9. Multiplicación")
        print("10. División")
        print("11. Potencia")
        print("12. Raíz cuadrada")
        print("13. Salir de la calculadora")

        SELECCION = input("Seleccione una operación (1-13): ")

        # sección de operaciones con dos números
        if SELECCION in ['7', '8', '9', '10', '11', '12']:
            numero1 = float(input("Ingrese un número: "))
            numero2 = float(input("Ingrese otro número: "))

            if SELECCION == '7':
                resultado = numero1 + numero2
                operacion = "Suma"
            elif SELECCION == '8':
                resultado = numero1 - numero2
                operacion = "Resta"
            elif SELECCION == '9':
                resultado = numero1 * numero2
                operacion = "Multiplicación"
            elif SELECCION == '10':
                if numero2 != 0:
                    resultado = numero1 / numero2
                    operacion = "División"
                else:
                    resultado = "Error: División por cero no permitida"
            elif SELECCION == '11':
                resultado = numero1 ** numero2
                operacion = "Potencia"
            elif SELECCION == '12':
                if numero1 < 0:
                    resultado = "Error: Raíz cuadrada de número negativo no permitida"
                else:
                    resultado = math.sqrt(numero1)
                    operacion = "Raíz cuadrada"

            print(f"{operacion} de {numero1} y {numero2} es: {resultado}")
        # Sección de operaciones trigonométricas
        elif SELECCION in ['1', '2', '3', '4', '5', '6']:
            numero = float(input("Ingrese un número (en grados): "))
            radianes = math.radians(numero)  # Convertir grados a radianes

            if SELECCION == '1':
                resultado = math.sin(radianes)
                operacion = "Seno"
            elif SELECCION == '2':
                resultado = math.cos(radianes)
                operacion = "Coseno"
            elif SELECCION == '3':
                resultado = math.tan(radianes)
                operacion = "Tangente"
            elif SELECCION == '4':
                resultado = 1 / math.tan(radianes)
                operacion = "Cotangente"
            elif SELECCION == '5':
                resultado = 1 / math.cos(radianes)
                operacion = "Secante"
            elif SELECCION == '6':
                resultado = 1 / math.sin(radianes)
                operacion = "Cosecante"

            print(f"{operacion} de {numero} grados es: {resultado}")
        # Sección de salida
        elif SELECCION == '13':
            print("Saliendo de la calculadora. ¡Hasta luego!")
            break
        else:
            print("Selección inválida. Por favor, intente de nuevo.")
    
calculadora()
#OPERACIONES CON DOS NÚMEROS
#numero2 = float(input("Ingrese otro numero: "))
#suma = numero1 + numero2
#print(f"La suma de {numero1} + {numero2} es: {suma}")
#resta = numero1 - numero2
#print(f"La resta de {numero1} - {numero2} es: {resta}")
#multiplicacion = numero1 * numero2
#print(f"La multiplicacion de {numero1} * {numero2} es: {multiplicacion}")
#division = numero1 / numero2
#print(f"La division de {numero1} / {numero2} es: {division}")   
#potencia = numero1 ** numero2
#print(f"La potencia de {numero1} ** {numero2} es: {potencia}")

#relacion de numeros
#if numero1 > numero2:
#    print(f"{numero1} es mayor que {numero2}")
#elif numero1 < numero2:
#    print(f"{numero1} es menor que {numero2}")
#else:
#    print(f"{numero1} es igual a {numero2}")
    