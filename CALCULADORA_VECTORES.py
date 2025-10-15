"""calculadora de vectores con menu
"""
import math

class vector():
    """
    Implementación de un vector
    """
    def __init__(self, x:float, y:float, z:float=None):
        self.x = x
        self.y = y
        if z is not None:
            self.z = z
            self.tresde = True
        else:
            self.z = None
            self.tresde = False
        self.modulo = abs(self) #define el valor absoluto al vector que estamos trabajando
    def phase(self):
        """Ángulo del vector en radianes"""
        if self.tresde:
            raise ValueError("Phase is not defined for 3D vectors") #si es 3d no se puede calcular el angulo
        else:
            return math.atan2(self.y, self.x)
    def __str__(self):
        """Representación en cadena del vector"""
        if self.tresde:
            respuesta = f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})" #: es para abrir la configuracion del dato
        else:
            respuesta = f"({self.x:.2f}, {self.y:.2f})" #:.2f me dice que coja dos decimales del valor float
        return respuesta
    def __abs__(self):
        """Norma Euclidea del vector"""
        if self.tresde:
            return pow(self.x**2 + self.y**2 + self.z**2, 0.5) # norma Euclidea
        else:
            return pow(self.x**2 + self.y**2, 0.5)# norma Euclidea
    def __add__(self, other):
        """Suma de vectores"""
        if self.tresde and other.tresde:
            return vector(self.x + other.x, self.y + other.y, self.z + other.z)
        elif not self.tresde and not other.tresde:
            return vector(self.x + other.x, self.y + other.y)
        else:
            raise ValueError("Cannot add 2D and 3D vectors together")
    def __sub__(self, other):
        """Resta de vectores"""
        if self.tresde and other.tresde:
            return vector(self.x - other.x, self.y - other.y, self.z - other.z)
        elif not self.tresde and not other.tresde:
            return vector(self.x - other.x, self.y - other.y)
        else:
            raise ValueError("No se puede restar un vector 2D y uno 3D")
    def __mul__(self, escalar):
        """Multiplicación por escalar"""
        if self.tresde:
            return vector(self.x * escalar, self.y * escalar, self.z * escalar)
        else:
            return vector(self.x * escalar, self.y * escalar)
    def __rmul__(self, escalar):
        """Multiplicación por escalar (orden invertido)"""
        return self.__mul__(escalar)
    def producto_cruz(self, other):
        """Producto vectorial (cruz) - solo para vectores 3D"""
        if not (self.tresde and other.tresde):
            raise ValueError("El producto vectorial solo está definido para vectores en 3D")
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return vector(x, y, z)
    def producto_punto(self, other):
        """Producto punto (escalar)"""
        if self.tresde and other.tresde:
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif not self.tresde and not other.tresde:
            return self.x * other.x + self.y * other.y
        else:
            raise ValueError("No se puede calcular el producto punto entre un vector 2D y uno 3D")

    def __ge__(self, other):
        """
        Compara la magnitud de dos vectores.
        """
        # Uso el método abs definido arriba usando norma Euclidea
        return abs(self) >= abs(other)


nombre = input("Ingrese su nombre: ")
apellido = input("Ingrese su apellido: ")
print("\nHola " + nombre.strip().lower().capitalize() + " " + apellido.strip().lower().capitalize() + " Bienvenido a la Calculadora de Vectores")

while True:
    print("\nMenú:")
    print("\n1. Ingresar Vectores")
    print("2. Calcular Suma")
    print("3. Calcular Resta")
    print("4. Calcular Multiplicación Escalar")
    print("5. calcular multiplicacion vectorial")
    print("6. Calcular Magnitud")
    print("7. Calcular Ángulo")
    print("8. Salir")

    opcion = input("Seleccione una opción: ")
    if opcion == "1":
        v1 = vector(*map(float, input("Ingrese el Vector 1 (x,y,z) y separados por comas: ").split(",")))
        v2 = vector(*map(float, input("Ingrese el Vector 2 (x,y,z) y separados por comas: ").split(",")))
        print("Los vectores ingresados son:")
        print(v1)
        print(v2)

    elif opcion == "2":
        print("\nLa suma de los vectores es:")
        print(v1 + v2)
    elif opcion == "3":
        print("\nLa resta de los vectores es:")
        # Implementación de resta de vectores
        if v1.tresde and v2.tresde:
            print(vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z))
        elif not v1.tresde and not v2.tresde:
            print(vector(v1.x - v2.x, v1.y - v2.y))
        else:
            print("No se puede restar un vector 2D y uno 3D.")
    elif opcion == "4":
        escalar = float(input("\nIngrese el escalar para multiplicar: "))
        print(f"La multiplicación del vector 1 por {escalar} es:")
        print(vector(v1.x * escalar, v1.y * escalar, v1.z * escalar if v1.tresde else None))
        print(f"La multiplicación del vector 2 por {escalar} es:")
        print(vector(v2.x * escalar, v2.y * escalar, v2.z * escalar if v2.tresde else None))
    elif opcion == "5":
        if v1.tresde and v2.tresde:
            cross_product = vector(
                v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x
            )
            print("El producto vectorial de los vectores es:")
            print(cross_product)
        else:
            print("El producto vectorial solo está definido para vectores en 3D.")
    elif opcion == "6":
        print("La magnitud de los vectores es:")
        print(f"El vector 1 {v1} es: {abs(v1):.2f}")
        print(f"El vector 2 {v2} es: {abs(v2):.2f}")

    elif opcion == "7":
        print(f"El ángulo del vector 1 {v1} es: {v1.phase():.2f} radianes")

    elif opcion == "8":
        print("Saliendo de la calculadora. ¡Hasta luego!")
        break   

#v1 = vector(*map(float, input("Ingrese el Vector 1 (x,y,z) sin.tresdes: ").split(",")))
#v2 = vector(*map(float, input("Ingrese el Vector 2 (x,y,z) sin.tresdes: ").split(",")))

#print("Los vectores ingresados son:")
#print(v1)
#print(v2)

#if v1 >= v2:
#    print(f"La magnitud del vector 10 es mayor o igual que la magnitud del vector 2")
#    print(f"{v1} >= {v2} es: {v1 >= v2}")

#else:
#    print(f"La magnitud del vector 1 = {v1} es menor que la magnitud del vector 2 = {v2}")

#print(f"El ángulo del vector 1 {v1} es: {v1.phase():.2f} radianes")

#print("La suma de los vectores es:")
#print(v1+v2)

#print(f"El vector 1 {v1} es: {abs(v1):.2f}")
#print(f"El vector 2 {v2} es: {abs(v2):.2f}")
#print(f"El vector 1 + vector 2 = {v1+v2} es: {abs(v1+v2):.2f}"  )
#print("La suma de los vectores es:")
#print(v1+v2)

#print(f"El vector 1 {v1} es: {abs(v1):.2f}")
#print(f"El vector 2 {v2} es: {abs(v2):.2f}")
#print(f"El vector 1 + vector 2 = {v1+v2} es: {abs(v1+v2):.2f}"  )