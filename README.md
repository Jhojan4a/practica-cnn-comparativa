# Práctica de Laboratorio 05: Arquitecturas CNN y Análisis de Resultados (MNIST vs. Fashion-MNIST)

**Institución:** Universidad Andina del Cusco  
**Facultad:** Facultad de Ingeniería y Arquitectura  
**Escuela Profesional:** Ingeniería de Sistemas  
**Curso:** SIS070 - Inteligencia Artificial  
**Unidad:** Unidad II - Redes Neuronales Convolucionales  
**Tema:** Arquitecturas CNN y Análisis de Resultados

---

## 1. Estructura del Repositorio

El proyecto presenta la implementación y comparación de dos redes neuronales convolucionales (CNN), utilizando los datasets MNIST y Fashion-MNIST.

```text
practica-cnn-comparativa/
│
├── .gitignore
├── README.md
├── mnist_cnn.py
└── fashion_cnn.py
```

**Descripción de los archivos:**

- `mnist_cnn.py`: Implementación de una CNN para clasificar dígitos escritos a mano (MNIST).
- `fashion_cnn.py`: Implementación de una CNN para clasificar prendas de vestir (Fashion-MNIST).
- `README.md`: Documentación técnica, resultados y análisis comparativo.
- `.gitignore`: Exclusión de archivos temporales y del entorno virtual.

---

## 2. Entorno de Desarrollo y Reproducibilidad

Para ejecutar el proyecto se utilizó Python con TensorFlow, NumPy y Matplotlib.

### 2.1. Creación del entorno virtual

```powershell
python -m venv venv
```

### 2.2. Activación del entorno virtual

```powershell
.\venv\Scripts\Activate.ps1
```

### 2.3. Instalación de dependencias

```powershell
pip install tensorflow numpy matplotlib
```

### 2.4. Ejecución de los programas

Para ejecutar el experimento MNIST:

```powershell
python mnist_cnn.py
```

Para ejecutar el experimento Fashion-MNIST:

```powershell
python fashion_cnn.py
```

---

# PARTE 1: Clasificación de Dígitos Escritos a Mano (MNIST)

## 3. Inspección Visual del Dataset

El dataset MNIST contiene imágenes de dígitos manuscritos del 0 al 9, representados mediante imágenes en escala de grises de 28 × 28 píxeles.

Durante la inspección visual se generó una cuadrícula de 3 × 3 que permite observar nueve imágenes del conjunto de datos.

**Características observadas:**

- Cada imagen tiene dimensiones de 28 × 28 píxeles.
- Los dígitos aparecen representados en color blanco o gris sobre un fondo negro.
- Las imágenes presentan diferentes estilos de escritura, inclinaciones y grosores de trazos.
- Los valores de los píxeles se encuentran originalmente en un rango de 0 a 255.

**Importancia de la inspección:**

La inspección permite comprobar la estructura de las imágenes y observar las diferencias entre los dígitos. Posteriormente, los valores de los píxeles se normalizan dividiéndolos entre 255.0, obteniendo valores comprendidos entre 0 y 1.

Esta normalización facilita el procesamiento de los datos durante el entrenamiento de la red neuronal.

## 4. Inspección de la Arquitectura mediante model.summary()

La arquitectura CNN implementada para MNIST contiene dos capas convolucionales, dos capas de agrupamiento MaxPooling, una capa Flatten, una capa densa oculta de 64 neuronas y una capa de salida Softmax de 10 neuronas.

### 4.1. Resumen de las capas

| Capa | Tipo | Output Shape | Parámetros |
|---|---|---|---:|
| Conv_1 | Conv2D | (None, 26, 26, 32) | 320 |
| MaxPool_1 | MaxPooling2D | (None, 13, 13, 32) | 0 |
| Conv_2 | Conv2D | (None, 11, 11, 64) | 18,496 |
| MaxPool_2 | MaxPooling2D | (None, 5, 5, 64) | 0 |
| Flatten | Flatten | (None, 1600) | 0 |
| Capa_Densa_Oculta | Dense | (None, 64) | 102,464 |
| Salida_Softmax | Dense | (None, 10) | 650 |

**Resumen de parámetros:**

- Total de parámetros: 121,930.
- Parámetros entrenables: 121,930.
- Parámetros no entrenables: 0.

### 4.2. Justificación geométrica de las dimensiones

**Primera convolución (Conv_1)**

La imagen original tiene dimensiones de 28 × 28 píxeles. La primera capa convolucional utiliza filtros de 3 × 3, un stride de 1 y `padding='valid'`.

La fórmula para calcular la dimensión de salida es:

\[
D_{salida}=\frac{D_{entrada}-K+2P}{S}+1
\]

Donde:

- D: dimensión de entrada.
- K: tamaño del kernel.
- P: cantidad de padding.
- S: tamaño del stride.

Reemplazando:

\[
D_{salida}=\frac{28-3+0}{1}+1=26
\]

Por lo tanto, la salida tiene dimensiones de 26 × 26 × 32, donde 32 representa la cantidad de filtros utilizados.

**Primera agrupación (MaxPool_1)**

Se utiliza una ventana de agrupamiento de 2 × 2 con un stride de 2.

\[
D_{salida}=\left\lfloor\frac{26}{2}\right\rfloor=13
\]

La salida resultante tiene dimensiones de 13 × 13 × 32.

La operación MaxPooling conserva el valor máximo de cada región de 2 × 2, reduciendo las dimensiones espaciales y manteniendo los 32 canales.

**Segunda convolución (Conv_2)**

La segunda capa convolucional recibe 32 canales y utiliza 64 filtros de 3 × 3.

\[
D_{salida}=\frac{13-3}{1}+1=11
\]

La salida es de 11 × 11 × 64.

**Segunda agrupación (MaxPool_2)**

Se vuelve a aplicar una ventana de 2 × 2 con stride de 2.

\[
D_{salida}=\left\lfloor\frac{11}{2}\right\rfloor=5
\]

La dimensión final es de 5 × 5 × 64.

## 5. Análisis de Parámetros y Función de Flatten

### 5.1. Cálculo de parámetros de Conv_2

Para calcular los parámetros entrenables de una capa convolucional se utiliza:

\[
P=(K_h \times K_w \times C_{entrada}\times F)+F
\]

Donde:

- Alto del kernel: 3.
- Ancho del kernel: 3.
- Canales de entrada: 32.
- Número de filtros: 64.
- Sesgos (bias): 64.

Reemplazando:

\[
P=(3\times3\times32\times64)+64
\]

\[
P=18,432+64=\boxed{18,496}
\]

Por lo tanto, la segunda capa convolucional contiene 18,496 parámetros entrenables.

### 5.2. Función de la capa Flatten

La capa Flatten transforma la salida tridimensional de MaxPool_2 en un vector unidimensional.

La entrada tiene dimensiones de:

\[
5\times5\times64=1,600
\]

Por lo tanto, Flatten convierte el tensor de 5 × 5 × 64 en un vector de 1,600 elementos.

Esta transformación permite conectar las capas convolucionales con las capas densas encargadas de realizar la clasificación.

Es importante mencionar que Flatten no realiza un aprendizaje de parámetros, únicamente reorganiza los datos.

## 6. Interpretación de los Resultados de MNIST

### 6.1. Resultados obtenidos en consola

Durante el entrenamiento se utilizaron 5,000 muestras durante una época.

```text
--- Entrenando el modelo MNIST (5000 muestras, 1 época) ---

79/79 ━━━━━━━━━━━━━━━━━━━━ 4s 28ms/step - accuracy: 0.7766 - loss: 0.8243

--- Analizando Resultados de Predicción (MNIST) ---

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step

Vector de probabilidades crudas (Softmax):
[0.000e+00 0.000e+00 0.000e+00 3.000e-04 0.000e+00
 0.000e+00 0.000e+00 9.991e-01 0.000e+00 6.000e-04]

Clase predicha por el modelo: 7
Clase real de la imagen: 7
```

### 6.2. Interpretación del vector Softmax

La función Softmax convierte los valores de salida de la red neuronal en una distribución de probabilidades.

La suma de las probabilidades de las diez clases es aproximadamente 1 (100 %).

En este experimento, el modelo asignó una probabilidad de 99.91 % al dígito 7.

Esto significa que la red neuronal identificó el dígito 7 como la clase con mayor probabilidad.

### 6.3. Función de np.argmax()

La función `np.argmax()` permite identificar la posición del valor máximo dentro del vector de probabilidades.

En este caso:

- Clase predicha: 7.
- Clase real: 7.
- Probabilidad asignada: 99.91 %.
- Resultado: clasificación correcta.

La predicción coincide con la etiqueta real de la imagen evaluada.

---

# PARTE 2: Clasificación de Prendas de Vestir (Fashion-MNIST)

## 7. Inspección Visual del Dataset

Fashion-MNIST es un conjunto de datos que contiene imágenes de prendas de vestir clasificadas en diez categorías diferentes.

Al igual que MNIST, cada imagen tiene dimensiones de 28 × 28 píxeles y utiliza una escala de grises.

Sin embargo, presenta una mayor complejidad visual debido a las diferentes formas, texturas y características de las prendas.

**Características observadas:**

- Las imágenes representan camisetas, pantalones, vestidos, abrigos, sandalias, zapatillas, bolsos y botines, entre otras categorías.
- Las prendas presentan diferentes siluetas y estructuras.
- Algunas categorías tienen características visuales similares.
- Las áreas de las prendas ocupan una parte importante de la imagen.

La inspección visual permite identificar la complejidad del conjunto de datos y comprender por qué su clasificación puede resultar más difícil que la clasificación de dígitos manuscritos.

## 8. Interpretación de class_names

Fashion-MNIST utiliza etiquetas numéricas del 0 al 9 para identificar las diferentes categorías de prendas.

Para facilitar la interpretación de los resultados se utiliza el arreglo `class_names`.

Este arreglo permite convertir el índice numérico obtenido por la red neuronal en el nombre correspondiente de la categoría.

Por ejemplo:

```python
class_names[9]
```

Resultado:

```text
Ankle boot
```

En español: Botín.

Por lo tanto, cuando el modelo predice el índice 9, se puede identificar que la prenda corresponde a la categoría Botín.

## 9. Análisis de la Arquitectura CNN de Fashion-MNIST

Para este experimento se mantuvo la estructura convolucional del modelo MNIST, pero se modificó la cantidad de neuronas de la capa densa oculta de 64 a 128.

### 9.1. Resumen de las capas

| Capa | Tipo | Output Shape | Parámetros |
|---|---|---|---:|
| Conv_1 | Conv2D | (None, 26, 26, 32) | 320 |
| MaxPool_1 | MaxPooling2D | (None, 13, 13, 32) | 0 |
| Conv_2 | Conv2D | (None, 11, 11, 64) | 18,496 |
| MaxPool_2 | MaxPooling2D | (None, 5, 5, 64) | 0 |
| Flatten | Flatten | (None, 1600) | 0 |
| Capa_Densa_Oculta | Dense | (None, 128) | 204,928 |
| Salida_Softmax | Dense | (None, 10) | 1,290 |

**Resumen de parámetros:**

- Total de parámetros: 225,034.
- Parámetros entrenables: 225,034.
- Parámetros no entrenables: 0.

### 9.2. Impacto de modificar la capa densa de 64 a 128 neuronas

La capa Flatten entrega un vector de 1,600 elementos a la capa densa oculta.

Para una arquitectura con 64 neuronas:

\[
P=(1600\times64)+64=102,464
\]

Para una arquitectura con 128 neuronas:

\[
P=(1600\times128)+128=204,928
\]

La capa de salida también aumenta sus parámetros debido a que recibe 128 entradas:

\[
P=(128\times10)+10=1,290
\]

En comparación con la arquitectura MNIST, el incremento total de parámetros es:

\[
225,034-121,930=\boxed{103,104}
\]

### 9.3. Interpretación del incremento de capacidad

El incremento de 64 a 128 neuronas permite que la capa densa tenga una mayor capacidad de representación y pueda aprender relaciones más complejas entre las características extraídas por las capas convolucionales.

Esto puede resultar útil para Fashion-MNIST, debido a que algunas prendas presentan similitudes visuales y requieren distinguir características específicas.

Sin embargo, una mayor cantidad de neuronas no garantiza automáticamente una mejor precisión. El rendimiento depende también de los datos, la cantidad de épocas, los hiperparámetros y el proceso de entrenamiento.

## 10. Interpretación de los Resultados de Fashion-MNIST

### 10.1. Vector de probabilidades obtenido

```text
[2.000e-04 1.000e-04 2.000e-03 6.000e-04 4.000e-04
 7.150e-02 1.300e-03 1.024e-01 2.470e-02 7.968e-01]
```

La red neuronal identificó la clase 9 como la categoría con mayor probabilidad.

**Interpretación de las probabilidades:**

| Categoría | Índice | Probabilidad |
|---|---:|---:|
| Sandalia | 5 | 7.15 % |
| Zapatilla | 7 | 10.24 % |
| Bolso | 8 | 2.47 % |
| Botín | 9 | 79.68 % |

El modelo asignó una probabilidad de 79.68 % a la categoría Botín, convirtiéndola en la predicción con mayor probabilidad.

### 10.2. Interpretación del resultado

La predicción muestra que el modelo pudo reconocer correctamente la prenda evaluada.

Sin embargo, también asignó probabilidades menores a otras categorías, principalmente Zapatilla y Sandalia.

Esto puede explicarse por las características visuales compartidas entre algunos tipos de calzado, como las suelas, los bordes y las siluetas.

La función Softmax permite observar estas diferencias de probabilidad y analizar cómo el modelo distribuye su confianza entre las distintas categorías.

---

# PARTE 3: Análisis Comparativo entre MNIST y Fashion-MNIST

## 11. Comparación de los Resultados de Entrenamiento

Los dos experimentos se ejecutaron utilizando 5,000 muestras de entrenamiento y una época.

Los resultados obtenidos fueron los siguientes:

| Métrica | MNIST | Fashion-MNIST |
|---|---:|---:|
| Muestras de entrenamiento | 5,000 | 5,000 |
| Épocas | 1 | 1 |
| Loss | 0.8243 | 1.0488 |
| Accuracy | 77.66 % | 63.52 % |
| Tiempo por paso | 28 ms | 9 ms |

### 11.1. Análisis de precisión

El modelo MNIST obtuvo una precisión de 77.66 %, mientras que Fashion-MNIST obtuvo 63.52 %.

La diferencia entre ambos resultados es:

\[
77.66-63.52=\boxed{14.14\text{ puntos porcentuales}}
\]

Esto demuestra que, bajo las condiciones de entrenamiento utilizadas, el modelo obtuvo un mejor resultado en la clasificación de dígitos manuscritos.

Una posible explicación es que las imágenes de MNIST presentan patrones relativamente más sencillos de distinguir, mientras que Fashion-MNIST contiene prendas con características visuales similares.

### 11.2. Análisis de la función Loss

La función `loss` mide el error cometido por el modelo durante el entrenamiento.

Los resultados fueron:

- MNIST: 0.8243.
- Fashion-MNIST: 1.0488.

Fashion-MNIST presentó una pérdida mayor, lo que coincide con su menor precisión en este experimento.

Esto indica que el modelo tuvo mayores dificultades para clasificar correctamente las imágenes de prendas de vestir.

### 11.3. Análisis del tiempo de ejecución

Los tiempos registrados fueron de 28 ms por paso para MNIST y 9 ms por paso para Fashion-MNIST.

Estos valores corresponden a las ejecuciones observadas y pueden variar dependiendo del hardware, la carga del sistema y las condiciones de ejecución.

Por lo tanto, no se puede concluir que un dataset sea intrínsecamente más rápido de procesar únicamente a partir de estos resultados.

## 12. Comparación de la Complejidad Visual

### 12.1. MNIST

Las imágenes contienen dígitos manuscritos con trazos relativamente definidos sobre un fondo negro.

Aunque existen diferencias en la escritura de cada persona, las características generales de los números permiten que la red neuronal identifique patrones representativos.

### 12.2. Fashion-MNIST

Las imágenes representan prendas de vestir con diferentes estructuras, formas y características internas.

Algunas categorías presentan similitudes visuales importantes. Por ejemplo:

- Camisetas, camisas y abrigos.
- Sandalias, zapatillas y botines.
- Vestidos y otras prendas superiores.

Estas similitudes incrementan la dificultad de clasificación y requieren que la CNN aprenda características más representativas.

## 13. Comparación de los Vectores Softmax

### 13.1. Vector de probabilidades de MNIST

```text
[0.000e+00 0.000e+00 0.000e+00 3.000e-04 0.000e+00
 0.000e+00 0.000e+00 9.991e-01 0.000e+00 6.000e-04]
```

El modelo asignó el 99.91 % de probabilidad al dígito 7.

Esto demuestra que, para la imagen evaluada, la red neuronal concentró prácticamente toda su probabilidad en una sola categoría.

### 13.2. Vector de probabilidades de Fashion-MNIST

```text
[2.000e-04 1.000e-04 2.000e-03 6.000e-04 4.000e-04
 7.150e-02 1.300e-03 1.024e-01 2.470e-02 7.968e-01]
```

El modelo asignó el 79.68 % de probabilidad a la categoría Botín.

También distribuyó probabilidades menores entre otras categorías, principalmente Zapatilla (10.24 %) y Sandalia (7.15 %).

Esto permite observar que, para esta imagen específica, el modelo presentó mayor incertidumbre entre algunas categorías visualmente relacionadas.

### 13.3. Conclusión del análisis Softmax

La comparación de los vectores permite observar diferencias en la distribución de probabilidades de ambos modelos.

En MNIST, la predicción evaluada presentó una concentración de probabilidad muy alta en el dígito 7.

En Fashion-MNIST, la predicción del Botín también fue correcta, pero presentó una distribución más amplia entre otras categorías.

Es importante señalar que estos vectores corresponden a predicciones individuales y no representan por sí solos el comportamiento general de todos los datos de prueba.

---

# PARTE 4: Conclusiones Generales

## 14. Conclusiones

1. **Implementación de arquitecturas CNN:** Se logró implementar dos redes neuronales convolucionales utilizando TensorFlow y Keras para realizar tareas de clasificación de imágenes. Ambas arquitecturas utilizan capas convolucionales, MaxPooling, Flatten y capas densas.

2. **Importancia de las capas convolucionales:** Las capas Conv2D permiten extraer características visuales de las imágenes, mientras que MaxPooling reduce las dimensiones espaciales y conserva información relevante.

3. **Función de Flatten:** Se comprobó que esta capa permite transformar los mapas de características en un vector unidimensional para conectarlos con las capas densas.

4. **Diferencia de rendimiento entre datasets:** MNIST obtuvo una precisión de 77.66 %, mientras que Fashion-MNIST alcanzó 63.52 %. La diferencia observada fue de 14.14 puntos porcentuales, lo que evidencia una mayor dificultad de clasificación en Fashion-MNIST bajo las condiciones utilizadas.

5. **Impacto de la cantidad de neuronas:** El incremento de 64 a 128 neuronas en la capa densa aumentó el número de parámetros entrenables de 121,930 a 225,034, proporcionando una mayor capacidad de representación al modelo.

6. **Interpretación de Softmax:** Se comprendió que Softmax transforma las salidas de la red en probabilidades y que `np.argmax()` permite seleccionar la categoría con mayor probabilidad.

7. **Importancia de la inspección visual:** La exploración inicial de los datasets permitió reconocer las diferencias entre los dígitos manuscritos y las prendas de vestir, facilitando la interpretación de los resultados obtenidos.

## 15. Recomendaciones

- Incrementar el número de épocas para evaluar la evolución del aprendizaje.
- Comparar los resultados de entrenamiento con los resultados de evaluación sobre el conjunto de prueba.
- Experimentar con diferentes cantidades de neuronas en las capas densas.
- Evaluar diferentes optimizadores y tasas de aprendizaje.
- Utilizar matrices de confusión para identificar las categorías que presentan mayores errores de clasificación.
- Analizar las predicciones incorrectas para comprender las limitaciones de cada arquitectura.

---

**Universidad Andina del Cusco**  
**Escuela Profesional de Ingeniería de Sistemas**  
**Curso: SIS070 - Inteligencia Artificial**  
**Práctica de Laboratorio 05: Arquitecturas CNN y Análisis de Resultados**