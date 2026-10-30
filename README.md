# Práctica de Laboratorio 05: Arquitecturas CNN y Análisis de Resultados (MNIST vs. Fashion-MNIST)

## PARTE 1: Práctica con Dígitos Escritos a Mano (MNIST)

### 1. Inspección Visual del Dataset
Al ejecutar la visualización con Matplotlib, se observa que los dígitos manuscritos están centrados, representados en escala de grises con trazos claros sobre fondos predominantemente negros o nulos. Visualizar los datos crudos antes de introducirlos a la CNN es fundamental para:
- Confirmar el rango dinámico de intensidades y la resolución espacial ($28 \times 28$).
- Detectar la variabilidad de estilos tipográficos, grosores e inclinaciones en los trazos manuales.
- Comprobar que no existan artefactos, desalineaciones severas o ruido de fondo que requieran etapas adicionales de preprocesamiento o binarización.

### 2. Inspección de `model.summary()` y Shapes
- **Reducción de $28 \times 28$ a $26 \times 26$ en `Conv_1`**:  
  Al aplicar convolución con un kernel de tamaño $K = 3$, paso (stride) $S = 1$ y sin relleno (`padding='valid'`), la dimensión espacial de salida se calcula mediante:
  $$\text{Dim}_{\text{salida}} = \frac{W - K + 2P}{S} + 1 = \frac{28 - 3 + 0}{1} + 1 = 26$$
  El filtro no puede sobrepasar los bordes de la imagen, perdiéndose 1 píxel en cada extremo.
- **Reducción de `MaxPool_1` a la mitad**:  
  La capa utiliza `pool_size=(2, 2)` con stride predeterminado de 2. Esto toma ventanas disjuntas de $2 \times 2$ y conserva el valor máximo de cada una, reduciendo tanto el ancho como la altura a la mitad ($\lfloor 26 / 2 \rfloor = 13$), produciendo una salida con dimensiones $(13, 13, 32)$.

### 3. Análisis de Parámetros y Función de Flatten
- **Cálculo de parámetros entrenables en `Conv_2`**:  
  - Tamaño del kernel: $3 \times 3 = 9$  
  - Canales de entrada (mapas recibidos de `MaxPool_1`): $32$  
  - Filtros (mapas de características generados): $64$  
  - Cada filtro cuenta con un sesgo (bias): $1 \times 64 = 64$  
  $$\text{Parámetros} = (3 \times 3 \times 32 \times 64) + 64 = 18\,432 + 64 = 18\,496$$
- **Función de la capa `Flatten`**:  
  La salida de `MaxPool_2` tiene forma $(5, 5, 64)$, correspondiente a un tensor tridimensional por muestra. Las capas densas (fully connected) requieren vectores unidimensionales ($5 \times 5 \times 64 = 1\,600$). `Flatten` no introduce parámetros entrenables; actúa como un puente estructural que linealiza la información espacial extraída por los filtros convolucionales para alimentar a la capa densa oculta.

### 4. Interpretación de Resultados en Consola
- **Evidencia en consola**:
  ```plaintext
  Vector de probabilidades crudas (Softmax): 
  [0.000e+00 0.000e+00 0.000e+00 3.000e-04 0.000e+00 0.000e+00 0.000e+00 9.991e-01 0.000e+00 6.000e-04]
  Clase predicha por el modelo: 7
  Clase real de la imagen: 7
  ```
- **Vector de salida Softmax**: Softmax normaliza las activaciones lineales (logits) en una distribución de probabilidades donde cada elemento cumple $0 \le P(y=i\vert{}X) \le 1$. La suma de los 10 valores es idénticamente **$1.0$**, representando la certeza total distribuida en el espacio de clases mutuamente excluyentes.
- **Función `np.argmax()`**: Evalúa el vector de salida y extrae el índice que contiene la probabilidad más alta. En este caso devolvió el índice **`7`**, coincidiendo con la etiqueta real `y_test[0] = 7` con una certidumbre de **99.91%** (`9.991e-01`).

---

## PARTE 2: Práctica con Clasificación de Prendas (Fashion-MNIST)

### 1. Inspección Visual del Dataset
A diferencia de MNIST, donde la información se reduce a trazos lineales sobre fondo plano, Fashion-MNIST presenta siluetas con texturas internas continuas, pliegues, suelas, cuellos, mangas y variaciones de densidad de píxeles en áreas rellenas (e.g., camisetas vs. abrigos). Esto eleva la dimensionalidad y variabilidad intrínseca de los patrones que los filtros convolucionales deben aprender a extraer.

### 2. Análisis de la Salida Nominal (`class_names`)
La salida cruda del modelo es un índice entero de $0$ a $9$. El arreglo `class_names` mapea posicionalmente cada índice discreto con su categoría en lenguaje natural (`class_names[indice]`), permitiendo interpretar de inmediato que el índice 9 corresponde a **"Botín"** en lugar de manejar códigos abstractos.

### 3. Impacto de la Capa Densa Modificada (128 vs 64 Neuronas)
- **Incremento en parámetros**:  
  La entrada a la capa densa desde `Flatten` es un vector de $1\,600$ dimensiones.
  - Con 64 neuronas: $(1\,600 \times 64) + 64 = 102\,464$ parámetros.
  - Con 128 neuronas: $(1\,600 \times 128) + 128 = 204\,928$ parámetros.  
  Se agregan $102\,464$ pesos adicionales, elevando los parámetros totales del modelo a $225\,034$.
- **Implicancia en el modelado**: Las prendas de vestir contienen combinaciones de rasgos morfológicos más complejas que los trazos de los números. Ampliar la capa a 128 unidades incrementa la capacidad representacional del clasificador, permitiendo proyectar patrones de mayor abstracción sin saturar la capa intermedia.

---

## PARTE 3: Análisis Comparativo entre Ambos Algoritmos y Datasets

### 1. Comparativa de Métricas de Entrenamiento (Evidencia Real)
- **MNIST**: `accuracy: 0.7766` | `loss: 0.8243` (Tiempo: 4s, 28ms/step).
- **Fashion-MNIST**: `accuracy: 0.6352` | `loss: 1.0488` (Tiempo: 1s, 9ms/step).
- **Análisis**: En una sola época con 5,000 muestras, la red aprende con mayor velocidad en MNIST (+14.14% de accuracy frente a Fashion-MNIST). Esto ocurre porque la frontera de decisión entre dígitos numéricos es más nítida gracias al alto contraste entre el trazo y el fondo. Por el contrario, Fashion-MNIST sufre mayor ambigüedad visual en etapas iniciales.

### 2. Complejidad Visual y de Patrones
- **Morfología**: Los dígitos son líneas delgadas con baja ocupación espacial y fondo nulo. Las prendas ocupan casi la totalidad del encuadre de $28 \times 28$ píxeles y comparten formas globales (siluetas de torso para camisas/abrigos o bases alargadas para calzado).
- **Impacto en Softmax**: En MNIST, la simplicidad de los bordes permite al clasificador concentrar la probabilidad de forma casi monolítica. En Fashion-MNIST, las semejanzas estructurales provocan que la activación se distribuya entre múltiples categorías con geometrías parecidas.

### 3. Distribución de Confianza del Modelo
- **Vector MNIST**: `[0, 0, 0, 3e-4, 0, 0, 0, 0.9991, 0, 6e-4]`. La clase ganadora (7) acumuló el **99.91%** de confianza, con probabilidades residuales prácticamente nulas en las restantes clases.
- **Vector Fashion-MNIST**: `[..., 0.0715, 0.0013, 0.1024, 0.0247, 0.7968]`. Aunque identificó correctamente la clase real (**Botín**, índice 9 con **79.68%**), el vector evidencia dispersión: asignó un **10.24%** a Zapatilla (índice 7) y un **7.15%** a Sandalia (índice 5).  
Esto demuestra de manera empírica cómo el dominio de los datos y el solapamiento interclase impactan directamente en la dispersión de la confianza de la capa Softmax.