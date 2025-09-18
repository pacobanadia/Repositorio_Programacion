#calculadora de vectores

nombre = input("Ingrese su nombre: ")
apellido = input("Ingrese su apellido: ")
print("Hola " + nombre.strip().lower().capitalize() + " " + apellido.strip().lower().capitalize() + " Bienvenido a la Calculadora de Vectores")

class vector():
    """
    Implementación de un vector
    """
    def __init__(self, x:float, y:float, z:float=None):
        self.x = x
        self.y = y
        if z is not None:
            self.z = z
            self.Espacio = True
        else:
            self.z = None
            self.Espacio = False
        self.modulo = abs(self) #define el valor absoluto al vector que estamos trabajando
    def phase(self):
        from math import atan2
        return(atan2(self.y, self.x))

    def __str__(self):
        if self.Espacio:
            respuesta = f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})" #: es para abrir la configuracion del dato
        else:
            respuesta = f"({self.x:.2f}, {self.y:.2f})" #:.2f me dice que coja dos decimales del valor float
        return(respuesta)
    def __abs__(self):
        if self.Espacio:
            return(pow(self.x**2 + self.y**2 + self.z**2, 0.5))
        else:
            return(pow(self.x**2 + self.y**2, 0.5))
    def __add__(self, other):
        if self.Espacio and other.Espacio:
            return vector(self.x + other.x, self.y + other.y, self.z + other.z)
        elif not self.Espacio and not other.Espacio:
            return vector(self.x + other.x, self.y + other.y)
        else:
            raise ValueError("Cannot add 2D and 3D vectors together")
        coordenadaZ# no necesariamente tiene que ser una suma, pq yo mismo puedo definir mi suma como quiera
        return(vector(self.x + other.x, self.y + other.y, self.z + other.z)) #self.x es el vector que estamos trabajando, y other es otro dato 
    def __ge__(self, other):
        """
        Compara la magnitud de dos vectores.
    """
        # Uso el método abs definido arriba usando norma Euclidea
        return abs(self) >= abs(other)
    


v1 = vector(*map(float, input("Ingrese el Vector 1 (x,y,z) sin espacios: ").split(",")))
v2 = vector(*map(float, input("Ingrese el Vector 2 (x,y,z) sin espacios: ").split(",")))

print("Los vectores ingresados son:")
print(v1)
print(v2)

if v1 >= v2:
    print(f"La magnitud del vector 10 es mayor o igual que la magnitud del vector 2")
    print(f"{v1} >= {v2} es: {v1 >= v2}")

else:
    print(f"La magnitud del vector 1 = {v1} es menor que la magnitud del vector 2 = {v2}")


print(f"El ángulo del vector 1 {v1} es: {v1.phase():.2f} radianes")


print("La suma de los vectores es:")
print(v1+v2)

print(abs(v1+v2))

#v1 = vector(2,3,1)
#v2 = vector(4,5,6)