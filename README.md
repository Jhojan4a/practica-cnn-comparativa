# Práctica de Laboratorio 05: Arquitecturas CNN y Análisis de Resultados (MNIST vs. Fashion-MNIST)

## PARTE 1: Práctica con Dígitos Escritos a Mano (MNIST)

### 1. Inspección Visual del Dataset
Al ejecutar la cuadrícula de Matplotlib, se observa que los dígitos manuscritos están centrados, representados en escala de grises con trazos claros sobre fondos predominantemente negros o nulos. Visualizar los datos crudos antes de introducirlos a la CNN es fundamental para:
- Confirmar el rango dinámico de intensidades y la resolución espacial ($28 \times 28$).
- Detectar variabilidad de estilos tipográficos o inclinaciones en los trazos.
- Comprobar que no existan artefactos, desalineaciones severas o ruido de fondo que requieran etapas adicionales de preprocesamiento o binarización.

### 2. Inspección de `model.summary()` y Shapes
- **Reducción de $28 \times 28$ a $26 \times 26$ en `Conv_1`**:
  Al aplicar convolución con un kernel de tamaño $K = 3$, paso (stride) $S = 1$ y sin padding (`padding='valid'`), la dimensión espacial de salida se calcula mediante:
  $$\text{Dim}_{\text{salida}} = \frac{W - K + 2P}{S} + 1 = \frac{28 - 3 + 0}{1} + 1 = 26$$
  El filtro no puede sobrepasar los bordes de la imagen, perdiéndose 1 píxel por cada extremo.
- **Reducción de `MaxPool_1` a la mitad**:
  La capa utiliza `pool_size=(2, 2)` con stride predeterminado de 2. Esto toma ventanas disjuntas de $2 \times 2$ y conserva el valor máximo de cada una, reduciendo tanto el ancho como la altura a la mitad ($\lfloor 26 / 2 \rfloor = 13$), produciendo una salida de dimensiones $(13, 13, 32)$.

### 3. Análisis de Parámetros y Función de Flatten
- **Cálculo de parámetros entrenables en `Conv_2`**:
  - Tamaño del kernel: $3 \times 3 = 9$
  - Canales de entrada (mapas recibidos de `MaxPool_1`): $32$
  - Filtros (mapas de características generados): $64$
  - Cada filtro cuenta con un sesgo (bias): $1 \times 64 = 64$
  $$\text{Parámetros} = (3 \times 3 \times 32 \times 64) + 64 = 18\,432 + 64 = 18\,496$$
- **Función de la capa `Flatten`**:
  La salida de `MaxPool_2` tiene forma $(5, 5, 64)$, que es un tensor tridimensional por cada muestra. Las capas densas clásicas (fully connected) requieren vectores unidimensionales como entrada ($5 \times 5 \times 64 = 1\,600$). `Flatten` no introduce parámetros entrenables; actúa como un puente dimensional que linealiza la información espacial extraída por las convoluciones para alimentar a la capa densa oculta.

### 4. Interpretación de Resultados en Consola
- **Vector de salida Softmax**: La función Softmax transforma las activaciones finales (logits) en una distribución de probabilidades donde cada elemento cumple $0 \le P(y=i|X) \le 1$. La suma de los 10 valores es idénticamente **$1.0$** (o $0.9999$ por redondeo de punto flotante), representando el espacio muestral completo de eventos mutuamente excluyentes.
- **Función `np.argmax()`**: Evalúa el vector de salida y devuelve el índice con el valor de probabilidad más alto. Dicho índice corresponde a la clase predicha por la red, el cual se compara directamente con el escalar `y_test[0]` para verificar si la predicción fue acertada.

---

## PARTE 2: Práctica con Clasificación de Prendas (Fashion-MNIST)

### 1. Inspección Visual del Dataset
A diferencia de MNIST, donde la información se limita a trazos lineales básicos (bordes y esquinas de líneas), Fashion-MNIST presenta siluetas con texturas internas continuas, pliegues, suelas, cuellos, mangas y variaciones de densidad de píxeles en áreas rellenas (e.g., camisetas vs. abrigos). Esto eleva la dimensionalidad intrínseca de los patrones que los filtros convolucionales deben aprender a detectar.

### 2. Análisis de la Salida Nominal (`class_names`)
La salida cruda del modelo produce únicamente índices numéricos enteros de $0$ a $9$. El arreglo `class_names` mapea posicionalmente cada índice numérico discreto con su significado semántico en lenguaje natural (`class_names[indice]`), permitiendo una interpretación inmediata y comprensible para el usuario o sistema cliente.

### 3. Impacto de la Capa Densa Modificada (128 vs 64 Neuronas)
- **Incremento en parámetros**:
  La entrada a la capa densa desde `Flatten` es un vector de $1\,600$ dimensiones.
  - Con 64 neuronas: $(1\,600 \times 64) + 64 = 102\,464$ parámetros.
  - Con 128 neuronas: $(1\,600 \times 128) + 128 = 204\,928$ parámetros.
  Esto agrega $102\,464$ pesos adicionales a la arquitectura.
- **Implicancia en el modelado**: Las prendas de vestir poseen combinaciones de características visuales más ricas y variadas que los dígitos. Ampliar la capa a 128 unidades aumenta la capacidad de representación (anchura) del clasificador para aprender combinaciones no lineales complejas sin saturar prematuramente la capa intermedia.

---

## PARTE 3: Análisis Comparativo entre Ambos Algoritmos y Datasets

### 1. Comparativa de Métricas de Entrenamiento
- **MNIST**: Tras 1 época de entrenamiento con 5,000 muestras, la precisión suele rondar entre el 88% y el 94%, con una pérdida inferior a 0.40.
- **Fashion-MNIST**: En las mismas condiciones (1 época, 5,000 muestras), la precisión ronda entre el 70% y el 78%, con una pérdida superior a 0.65.
- **Razón**: MNIST converge con muchísima mayor rapidez debido a que la separación interclase es amplia (el fondo negro sin ruido hace que los bordes de los números sean fáciles de discriminar con pocos filtros iniciales). Fashion-MNIST presenta solapamientos morfológicos significativos (por ejemplo, diferenciar una camiseta de una camisa o un abrigo exige detectar patrones locales sutiles como cuellos o botones).

### 2. Complejidad Visual y de Patrones
- **Morfología**: Los dígitos manuscritos son trazos simples centrados en un fondo predominantemente nulo. Las prendas ocupan la mayor parte del encuadre espacial de $28 \times 28$ píxeles y contienen gradientes continuos de gris.
- **Impacto en Softmax**: Debido a la menor ambigüedad morfológica en MNIST, la red distribuye la probabilidad de manera casi unívoca hacia la clase correcta. En Fashion-MNIST, prendas de corte similar suelen compartir activación, diluyendo la masa de probabilidad entre varias clases candidatas.

### 3. Distribución de Confianza del Modelo
- En MNIST, el vector Softmax tiende a concentrar una probabilidad superior al 90% en la neurona ganadora, manteniendo las 9 restantes cercanas a 0.0000.
- En Fashion-MNIST, el vector crudo es notablemente más disperso: la clase predicha puede registrar entre 50% y 75% de certeza, mientras que clases morfológicamente afines (como Pulóver, Abrigo y Camisa) acumulan porcentajes residuales apreciables (10% a 20%). Esto evidencia que la complejidad y entropía del dominio de datos incide de manera directa en la certidumbre de la red.