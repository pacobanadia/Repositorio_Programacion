print("hola"+"mundo") #concatenacion
print("hola"*4) #replica
print("hola"[-3]) #indexacion negativa
print("hola MUNDO"[2:7]) #slicing
print("hola MUNDO"[2:]) #slicing con step
print("hola MUNDO"[:7]) #slicing con step
print("HOLA MUNDO"[2:-2]) #slicing con indexacion negativa
print("HOLA MUNDO"[:]) #copia


nombre="juan"
apellido="perez"
nombre_completo=nombre+" "+apellido
print(nombre_completo)

print(f"{nombre} {apellido}")
print (f"nombre_completo:{nombre_completo}") #f-string
print("{0} {1}".format(nombre,apellido)) #metodo format
print("yo amo los {0} con {1}, de verdad los {0} y la {1} son lo mejor... {0}! {0}! {0}! ".format("tacos","cebolla")) #metodo format
print("yo deseo escribir 'hola'") #comillas simples
print("holan mundo") #caracter de escape
print("yo deseo escribir \"hola\"") #caracter de escape

print("hola\t mundo") #tabulacion
print("hola\b mundo") #backspace
print("hola\f mundo") #form feed
print("hola\v mundo") #vertical tab
print("hola\r mundo") #carriage return
print("hola\\ mundo") #backslash

print("hola\rh") #carriage return

nombre1="juan"
apellido1="perez"
edad1=30
altura1=1.75
nombre2="maria"
apellido2="lopez"
edad2=25
altura2=1.65
nombre3="carlos"
apellido3="garcia"
edad3=35
altura3=1.80
nombre4="ana"
apellido4="martinez"
edad4=28
altura4=1.70
nombre5="luis"
apellido5="gonzalez"
edad5=32
altura5=1.78

print(
    f"nombre:{nombre1} apellido:{apellido1} edad:{edad1} altura:{altura1}, "
    f"nombre:{nombre2} apellido:{apellido2} edad:{edad2} altura:{altura2}, "
    f"nombre:{nombre3} apellido:{apellido3} edad:{edad3} altura:{altura3}, "
    f"nombre:{nombre4} apellido:{apellido4} edad:{edad4} altura:{altura4}, "
    f"nombre:{nombre5} apellido:{apellido5} edad:{edad5} altura:{altura5}"
)
print(" aDolFo ".strip()) #elimina espacios al inicio y al final
print(" aDolFo ".lstrip()) #elimina espacios al inicio
print(" aDolFo ".rstrip()) #elimina espacios al final
print("aDolFo".lower()) #convierte a minusculas
print("aDolFo".upper()) #convierte a mayusculas
print("aDolFo".capitalize()) #convierte la primera letra a mayuscula
print("aDolFo".title()) #convierte la primera letra de cada palabra a mayusculas
print("aDolFo".count("o")) #cuenta las veces que aparece un caracter
print("aDolFo".find("o")) #busca la posicion de un caracter
print("aDolFo".replace("o","a")) #reemplaza un caracter por otro
print("aDolFo".startswith("a")) #verifica si empieza con un caracter
print("aDolFo".endswith("o")) #verifica si termina con un caracter
print("aDolFo".isalpha()) #verifica si es alfabetico
print("1234".isdigit()) #verifica si es numerico
print("aDolFo123".isalnum()) #verifica si es alfanumerico
print("r"in"roberto") #verifica si un caracter esta en la cadena
print(len("aDolFo")) #longitud de la cadena
