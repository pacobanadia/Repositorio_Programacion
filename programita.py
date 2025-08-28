# Programa que saluda al usuario por su nombre y apellido  

nombre = input("Ingrese su nombre: ")
apellido = input("Ingrese su apellido: ")
print("Hola " + nombre.strip().lower().capitalize() + " " + apellido.strip().lower().capitalize())

numero1 = float(input("Ingrese un numero: "))
numero2 = float(input("Ingrese otro numero: "))
suma = numero1 + numero2
print(f"La suma de {numero1} + {numero2} es: {suma}")
resta = numero1 - numero2
print(f"La resta de {numero1} - {numero2} es: {resta}")
multiplicacion = numero1 * numero2
print(f"La multiplicacion de {numero1} * {numero2} es: {multiplicacion}")
division = numero1 / numero2
print(f"La division de {numero1} / {numero2} es: {division}")   
potencia = numero1 ** numero2
print(f"La potencia de {numero1} ** {numero2} es: {potencia}")

#relacion de numeros
if numero1 > numero2:
    print(f"{numero1} es mayor que {numero2}")
elif numero1 < numero2:
    print(f"{numero1} es menor que {numero2}")
else:
    print(f"{numero1} es igual a {numero2}")
    