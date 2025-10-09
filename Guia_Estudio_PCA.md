# Guía de Estudio: Análisis de Componentes Principales (PCA)

## Tabla de Contenidos
1. [Introducción al PCA](#introducción-al-pca)
2. [Fundamentos Matemáticos](#fundamentos-matemáticos)
3. [¿Por qué PCA? Motivación](#por-qué-pca-motivación)
4. [Tipo de Datos: Discretos vs Continuos](#tipo-de-datos-discretos-vs-continuos)
5. [Análisis del Código Paso a Paso](#análisis-del-código-paso-a-paso)
6. [Selección de Componentes Principales](#selección-de-componentes-principales)
7. [Interpretación de Resultados](#interpretación-de-resultados)
8. [Sparse PCA](#sparse-pca)
9. [Ejercicios de Estudio](#ejercicios-de-estudio)

---

## Introducción al PCA

### ¿Qué es PCA?

El **Análisis de Componentes Principales (PCA)** es una técnica de reducción de dimensionalidad que transforma un conjunto de variables posiblemente correlacionadas en un conjunto más pequeño de variables no correlacionadas llamadas **componentes principales**.

### Objetivos del PCA:
1. **Reducir dimensionalidad**: Pasar de `p` variables a `k` componentes (donde `k < p`)
2. **Eliminar redundancia**: Remover correlaciones entre variables
3. **Visualización**: Facilitar la visualización de datos de alta dimensión
4. **Reducción de ruido**: Conservar solo la información más relevante

---

## Fundamentos Matemáticos

### 1. Matriz de Datos

Consideremos una matriz de datos **X** de dimensión `n × p`:
- `n` = número de observaciones
- `p` = número de variables

```
X = [x₁, x₂, ..., xₚ]  donde cada xᵢ es un vector de n observaciones
```

### 2. Matriz de Covarianza

La matriz de covarianza **Σ** (o **S** para la muestra) es:

```
S = (1/(n-1)) · Xᵀ · X  (si X está centrada)
```

**Elementos de la matriz:**
- Diagonal: Varianzas de cada variable
- Fuera de diagonal: Covarianzas entre pares de variables

**¿Por qué es importante?**
- Mide cómo las variables varían conjuntamente
- PCA busca las direcciones de máxima varianza en esta matriz

### 3. Valores y Vectores Propios

**Problema de eigenvalores:**
```
S · v = λ · v
```

Donde:
- **λ** (lambda) = valor propio (eigenvalue)
- **v** = vector propio (eigenvector)

**En PCA:**
- Los **vectores propios** definen las direcciones de los componentes principales
- Los **valores propios** indican la varianza explicada por cada componente

### 4. Descomposición Espectral

La matriz de covarianza S se puede descomponer como:

```
S = V · Λ · Vᵀ
```

Donde:
- **V** = matriz de vectores propios (columnas)
- **Λ** = matriz diagonal de valores propios
- **Vᵀ** = transpuesta de V

### 5. Componentes Principales

Las componentes principales son combinaciones lineales de las variables originales:

```
PC₁ = v₁₁·x₁ + v₁₂·x₂ + ... + v₁ₚ·xₚ
PC₂ = v₂₁·x₁ + v₂₂·x₂ + ... + v₂ₚ·xₚ
...
```

**Propiedades importantes:**
1. **PC₁** tiene la máxima varianza posible
2. **PC₂** tiene la máxima varianza posible ortogonal a PC₁
3. Y así sucesivamente...
4. Las PCs son **ortogonales** entre sí (no correlacionadas)

### 6. Scores vs Loadings

**Loadings (Cargas):**
- Son los coeficientes `vᵢⱼ` en las ecuaciones anteriores
- Son los vectores propios de la matriz de covarianza
- Indican cómo contribuye cada variable original a cada PC

**Scores (Puntuaciones):**
- Son los valores de las observaciones en el nuevo sistema de coordenadas
- Se calculan como: `Scores = X · V`
- Son las proyecciones de los datos en las direcciones de las PCs

---

## ¿Por qué PCA? Motivación

### Problema: Maldición de la Dimensionalidad

Cuando tenemos muchas variables (`p` grande):
- Visualización difícil (solo podemos ver en 2D o 3D)
- Modelos complejos y sobreajuste
- Variables redundantes (correlacionadas)
- Mayor costo computacional

### Solución: PCA

**Ejemplo conceptual:**

Imagina que tienes datos sobre cereales con 13 variables:
```
calories, protein, fat, sodium, fiber, carbo, sugars, potass, 
vitamins, shelf, weight, cups, rating
```

Muchas de estas variables están correlacionadas:
- **Calorías** ↔ Carbohidratos (alta correlación positiva)
- **Fibra** ↔ Rating (correlación positiva)
- **Azúcar** ↔ Rating (correlación negativa)

**PCA encuentra:**
- PC₁ podría representar "contenido energético general"
- PC₂ podría representar "salud nutricional"
- PC₃ podría representar "contenido de vitaminas"

En lugar de 13 variables, podemos trabajar con 2-3 componentes que capturan el 80-90% de la información.

---

## Tipo de Datos: Discretos vs Continuos

### ¿PCA funciona con datos discretos o continuos?

**Respuesta corta:** PCA está diseñado principalmente para **variables continuas**.

### Variables Continuas ✅

**Por qué funciona bien:**
1. Las variables continuas tienen varianza continua
2. Las distancias euclidianas tienen sentido
3. La covarianza es interpretable
4. Ejemplos: altura, peso, temperatura, precio

**En nuestro código:**
```python
cereals_df[["calories", "rating"]]
```
- `calories`: continua (50, 100, 110, etc.)
- `rating`: continua (escala de evaluación)

### Variables Discretas ⚠️

**Precaución con:**
1. **Variables categóricas nominales** (rojo, azul, verde)
   - NO usar PCA directamente
   - Necesitan codificación especial (dummy variables)
   
2. **Variables ordinales** (bajo, medio, alto)
   - Se puede usar si hay suficientes categorías
   - Asumir distancias iguales entre categorías
   
3. **Variables de conteo** (0, 1, 2, 3, ...)
   - Se puede usar si el rango es amplio
   - Considerar transformaciones (log, sqrt)

### ¿Por qué la distinción es importante?

**Matemáticamente:**
- PCA calcula distancias euclidianas: `d = √((x₁-y₁)² + (x₂-y₂)² + ...)`
- Para variables continuas, estas distancias son significativas
- Para categóricas, "rojo" - "azul" no tiene sentido matemático

**Ejemplo:**
```
Variables continuas:
  Altura: 170 cm vs 175 cm → diferencia = 5 cm (interpretable)
  
Variables categóricas:
  Color: 1 (rojo) vs 2 (azul) → diferencia = 1 (sin sentido)
```

### Recomendaciones

**Usa PCA cuando:**
- ✅ Todas las variables son continuas
- ✅ Variables ordinales con muchas categorías (>5)
- ✅ Variables de conteo con rango amplio

**NO uses PCA cuando:**
- ❌ Variables categóricas nominales
- ❌ Variables binarias (0/1)
- ❌ Mezcla de tipos de variables sin preprocesamiento

**Alternativas:**
- **MCA** (Multiple Correspondence Analysis) para categóricas
- **FAMD** (Factor Analysis of Mixed Data) para mixtas

---

## Análisis del Código Paso a Paso

### Sección 1: PCA en Dos Variables (Calorías y Rating)

```python
cereals_df = pd.read_csv("/content/Cereals.csv")
pcs = PCA(n_components=2)
pcs.fit(cereals_df[["calories", "rating"]])
```

**¿Por qué estas dos variables?**

1. **Propósito didáctico**: Con 2 variables podemos visualizar en 2D
2. **Posible correlación**: Exploramos si hay relación entre calorías y rating
3. **Interpretación simple**: Fácil de entender para aprendizaje

**Matemáticamente:**
- X es una matriz `n × 2`
- Solo habrá 2 componentes principales posibles
- min(n, p) = min(n, 2) = 2

### Resumen de PCA (pcsSummary)

```python
pcsSummary = pd.DataFrame({
    "Standard deviation": np.sqrt(pcs.explained_variance_),
    "Proportion of variance": pcs.explained_variance_ratio_,
    "Cumulative proportion": np.cumsum(pcs.explained_variance_ratio_)
})
```

**Interpretación de cada métrica:**

1. **Standard deviation (Desviación estándar)**
   ```
   SD_i = √(λᵢ)
   ```
   - Es la raíz cuadrada del valor propio i
   - Mide la dispersión de los datos en la dirección de PCᵢ
   - Mayor SD → más información capturada

2. **Proportion of variance (Proporción de varianza)**
   ```
   PropVar_i = λᵢ / Σλⱼ
   ```
   - Fracción de la varianza total explicada por PCᵢ
   - Valores entre 0 y 1
   - Suma de todas las proporciones = 1

3. **Cumulative proportion (Proporción acumulada)**
   ```
   CumProp_k = Σ(λᵢ / Σλⱼ)  para i=1 hasta k
   ```
   - Varianza total explicada hasta el componente k
   - Útil para decidir cuántos componentes retener

**Ejemplo de interpretación:**
```
         PC1    PC2
SD       15.2   8.3
PropVar  0.77   0.23
CumProp  0.77   1.00
```
Significa:
- PC1 explica 77% de la varianza total
- PC2 explica 23% adicional
- Juntos explican 100% (porque solo hay 2 variables)

### Components (Loadings/Cargas)

```python
pcsComponents_df = pd.DataFrame(
    pcs.components_.transpose(), 
    columns=["PC1", "PC2"],
    index=["calories", "rating"]
)
```

**¿Qué representan?**

Matriz de vectores propios:
```
          PC1     PC2
calories  0.89   -0.45
rating    0.45    0.89
```

**Interpretación:**
- **PC1 = 0.89·calories + 0.45·rating**
  - Dominada por calorías (coeficiente mayor)
  - Ambas variables contribuyen positivamente
  - Representa algo como "intensidad nutricional"

- **PC2 = -0.45·calories + 0.89·rating**
  - Dominada por rating
  - Calorías contribuyen negativamente
  - Representa algo como "calidad vs cantidad"

**Propiedades matemáticas:**
1. Cada columna tiene norma 1: `√(0.89² + 0.45²) = 1`
2. Las columnas son ortogonales: `PC1 · PC2 = 0`

### Scores (Puntuaciones)

```python
scores = pd.DataFrame(
    pcs.transform(cereals_df[["calories", "rating"]]),
    columns=["PC1","PC2"]
)
```

**¿Qué son?**
- Son las coordenadas de cada observación (cereal) en el nuevo sistema
- Se calculan como: `Score_i = Xᵢ · V`

**Ejemplo:**
Si un cereal tiene:
- calories = 100
- rating = 50

Su score sería:
```
PC1_score = 0.89 × 100 + 0.45 × 50 = 89 + 22.5 = 111.5
PC2_score = -0.45 × 100 + 0.89 × 50 = -45 + 44.5 = -0.5
```

**Utilidad:**
- Cada fila = un cereal en el espacio de componentes principales
- Podemos plotear estos scores para visualizar patrones
- Detectar outliers
- Hacer clustering en espacio reducido

### Visualización: Scatterplot de Scores

```python
plt.plot(scores.PC1[:], scores.PC2[:], 'o')
```

**¿Qué observamos?**
- Cada punto = un cereal
- Posición en PC1 y PC2
- Patrones, clusters, outliers
- Relaciones no visibles en variables originales

---

### Sección 2: PCA con Escalamiento en Todas las Variables

```python
pcs = PCA()
pcs.fit(preprocessing.scale(cereals_df.iloc[:, 3:].dropna(axis=0)))
```

**Cambios importantes:**

1. **`PCA()` sin n_components**
   - Calcula TODAS las componentes posibles
   - min(n, p) componentes
   
2. **`preprocessing.scale()`**
   - ¡MUY IMPORTANTE!
   - Estandariza las variables: media=0, varianza=1
   
3. **`cereals_df.iloc[:, 3:]`**
   - Toma desde la columna 4 en adelante
   - Asume que las primeras 3 son identificadores o categóricas

### ¿Por qué escalar? 🔥 CRÍTICO

**Problema sin escalar:**

Imagina variables con diferentes unidades:
```
calories:  [50, 100, 150]     (rango: 100)
sodium:    [0, 200, 400]      (rango: 400)
fiber:     [1, 3, 5]          (rango: 4)
```

**PCA sin escalar:**
- Sodium dominaría (mayor varianza por escala)
- PC1 ≈ sodium casi completamente
- Información de fiber se perdería

**PCA con escalar:**
```
calories:  [-1.22, 0, 1.22]   (varianza: 1)
sodium:    [-1.22, 0, 1.22]   (varianza: 1)
fiber:     [-1.22, 0, 1.22]   (varianza: 1)
```
- Todas las variables tienen igual "peso"
- Las componentes reflejan correlaciones, no escalas

**Regla:**
- ✅ **SIEMPRE escalar** cuando las variables tienen diferentes unidades
- ⚠️ Puede no escalar si todas las variables están en la misma escala

**Matemáticamente:**
```python
x_scaled = (x - mean(x)) / std(x)
```

Esto convierte la matriz de covarianza en matriz de correlación:
```
Cov(X_scaled) = Corr(X)
```

### Visualización 3D

```python
ax = plt.axes(projection='3d')
ax.scatter3D(xline, yline, zline, cmap='Greens')
```

**Utilidad:**
- Visualizar datos en las 3 primeras PCs
- Ver estructura de datos en 3D
- Las 3 primeras PCs usualmente capturan 70-90% de varianza

---

### Sección 3: Sparse PCA

```python
spca = SparsePCA(random_state=0, alpha=1e-3, ridge_alpha=1e-6)
```

**¿Qué es Sparse PCA?**

**Problema del PCA estándar:**
- Los loadings (cargas) suelen ser TODOS no-cero
- Difícil interpretación: todas las variables contribuyen un poco

**Ejemplo PCA regular:**
```
PC1 = 0.23·cal + 0.18·prot + 0.31·fat + 0.22·sod + ...
```
Todas las variables aparecen, difícil de interpretar.

**Sparse PCA:**
- Fuerza muchos loadings a ser EXACTAMENTE cero
- Solo unas pocas variables por componente

**Ejemplo Sparse PCA:**
```
PC1 = 0.50·cal + 0·prot + 0.50·fat + 0·sod + ...
```
Solo calorías y grasa importan, ¡mucho más interpretable!

**Trade-off:**
- ✅ Más interpretable
- ✅ Menos variables por componente
- ❌ Pierde un poco de varianza explicada
- ❌ Más costoso computacionalmente

**Parámetros:**
- `alpha`: Controla el nivel de sparsity (mayor = más ceros)
- `ridge_alpha`: Regularización adicional para estabilidad

**Matemáticamente:**

PCA minimiza:
```
||X - X·V·Vᵀ||²
```

Sparse PCA minimiza:
```
||X - X·V·Vᵀ||² + α·||V||₁
```

Donde `||V||₁` es la norma L1 que promueve sparsity.

---

## Selección de Componentes Principales

### ¿Cuántos componentes retener?

**Pregunta clave:** De las `p` componentes calculadas, ¿cuántas usar?

### Criterio 1: Proporción de Varianza

**Regla del 80-90%:**
Retener suficientes componentes para explicar 80-90% de varianza total.

```python
cumsum = np.cumsum(pcs.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.80) + 1
```

**Ejemplo:**
```
PC1: 45%  → Acum: 45%
PC2: 28%  → Acum: 73%
PC3: 15%  → Acum: 88%  ← Retener hasta aquí
PC4: 8%   → Acum: 96%
PC5: 4%   → Acum: 100%
```

**Por qué:**
- Las primeras PCs capturan la estructura principal
- Las últimas PCs suelen ser ruido
- 80-90% es un balance razonable

### Criterio 2: Scree Plot

Graficar valores propios (o varianza explicada) vs número de componente.

```
Varianza |     •
         |      
         |       •
         |         
         |          •  •
         |___________•__•__•_
            1  2  3  4  5  6
         Componentes
```

**Buscar el "codo":**
- Donde la curva se aplana
- Antes del codo: información real
- Después del codo: ruido

### Criterio 3: Regla de Kaiser

**Retener componentes con valor propio > 1** (si datos están escalados)

**Justificación:**
- Si λᵢ > 1, PCᵢ explica más varianza que una variable original
- Si λᵢ < 1, PCᵢ explica menos que una variable original (no vale la pena)

### Criterio 4: Validación Cruzada

Probar diferentes valores de `k` en modelo posterior y evaluar performance.

```python
for k in range(1, p):
    X_reduced = pcs.transform(X)[:, :k]
    score = evaluate_model(X_reduced)
```

Elegir `k` con mejor score.

### ¿Por qué funcionan las primeras 2-3 componentes?

**Razones matemáticas:**

1. **Teorema de Descomposición Espectral**
   - Los valores propios están ordenados: λ₁ ≥ λ₂ ≥ ... ≥ λₚ
   - λ₁ es la dirección de máxima varianza
   - λ₂ es la segunda dirección de máxima varianza, etc.

2. **Ley de Pareto (80/20)**
   - En datos reales, pocas variables explican mucho
   - La mayor parte de variación está en pocas dimensiones

3. **Estructura de correlación**
   - Variables correlacionadas → información redundante
   - PCA elimina redundancia
   - Ejemplo: 10 variables con alta correlación → 2-3 PCs suficientes

4. **Maldición de la dimensionalidad**
   - Más dimensiones → más ruido
   - Dimensiones altas suelen ser ruido
   - Primeras PCs = señal, últimas PCs = ruido

**Analogía:**
Imagina comprimir una imagen:
- Primeros componentes = contornos principales, colores dominantes
- Últimos componentes = pixeles individuales, ruido
- Con 10% de componentes, recuperas 90% de la imagen

---

## Interpretación de Resultados

### Loadings (Cargas)

**¿Cómo interpretar?**

```
         PC1    PC2    PC3
calories  0.4    0.2   -0.1
protein   0.3    0.5    0.3
fat       0.5   -0.2    0.1
sugar    -0.4    0.3    0.6
fiber     0.3    0.6   -0.4
```

**Para PC1:**
- **Positivos altos** (fat: 0.5, calories: 0.4): "Altos en energía"
- **Negativos** (sugar: -0.4): "Bajos en azúcar"
- **Interpretación**: PC1 representa "Densidad energética sin azúcar"

**Magnitud de loadings:**
- |loading| > 0.4: Variable muy importante
- 0.2 < |loading| < 0.4: Moderadamente importante
- |loading| < 0.2: Poco importante

**Signo de loadings:**
- Positivo: Variable aumenta con el componente
- Negativo: Variable disminuye con el componente

### Scores

**¿Cómo interpretar?**

Un cereal con:
```
PC1_score = 3.5   (alto)
PC2_score = -1.2  (bajo)
PC3_score = 0.1   (neutro)
```

Interpretación basada en PC1 y PC2:
- Alto en PC1 → Alta densidad energética sin azúcar
- Bajo en PC2 → (interpretar según loadings de PC2)

**Utilidad:**
- Comparar cereales en espacio reducido
- Identificar grupos similares
- Detectar anomalías

### Biplot

Combina scores y loadings en un mismo gráfico:
- Puntos = observaciones (scores)
- Vectores = variables (loadings)

**Interpretación:**
- Puntos cercanos → observaciones similares
- Vectores largos → variables con alta varianza
- Vectores en misma dirección → variables correlacionadas positivamente
- Vectores en dirección opuesta → correlación negativa

---

## Sparse PCA

### Diferencias con PCA regular

| Aspecto | PCA Regular | Sparse PCA |
|---------|-------------|------------|
| Loadings | Todos ≠ 0 | Muchos = 0 |
| Interpretación | Difícil | Fácil |
| Varianza explicada | Máxima | Ligeramente menor |
| Costo computacional | Bajo | Alto |
| Uso | Reducción dimensión | Interpretación + reducción |

### Cálculo de varianza en Sparse PCA

```python
Q, R = np.linalg.qr(spca.components_)
R2 = np.diag(R) * np.diag(R)
```

**¿Por qué QR?**
- Sparse PCA no garantiza ortogonalidad perfecta
- Descomposición QR ortogonaliza los componentes
- R² captura la varianza de cada componente ortogonalizado

**Ordenamiento:**
```python
arr1 = np.argsort(R2)[::-1]
R2_ = R2[arr1]
```
- Ordena componentes por varianza (mayor a menor)
- Similar a PCA regular

---

## Ejercicios de Estudio

### Ejercicio 1: Conceptual

**Pregunta:** Tienes un dataset con 100 observaciones y 50 variables. Después de PCA:
- PC1 explica 60% de varianza
- PC2 explica 20%
- PC3 explica 10%
- Resto: 10% distribuido en PC4-PC50

¿Cuántos componentes retendrías y por qué?

**Análisis:**
- 3 componentes: 90% de varianza ✅
- Resto parece ruido (10% en 47 componentes)
- Criterio 80-90%: 2-3 componentes

### Ejercicio 2: Cálculo Manual

Dadas dos variables con matriz de covarianza:
```
S = | 4   2 |
    | 2   3 |
```

**Calcular:**
a) Valores propios (eigenvalues)
b) Vectores propios (eigenvectors)
c) Proporción de varianza de cada PC

**Solución:**
```
det(S - λI) = 0
(4-λ)(3-λ) - 4 = 0
λ² - 7λ + 8 = 0
λ₁ = 5.56, λ₂ = 1.44

PropVar₁ = 5.56/7 = 79.4%
PropVar₂ = 1.44/7 = 20.6%
```

### Ejercicio 3: Interpretación

Dataset de estudiantes con variables:
```
- horas_estudio
- horas_sueño
- nota_final
- faltas
- participación
```

Supongamos PC1 tiene loadings:
```
horas_estudio: 0.45
horas_sueño: 0.30
nota_final: 0.50
faltas: -0.40
participación: 0.35
```

**Interpretar PC1:**
- Positivo: buen estudiante (estudia, duerme bien, buena nota, participa)
- Negativo: mal estudiante (no estudia, falta)
- PC1 = "Índice de compromiso académico"

### Ejercicio 4: Escalamiento

**Pregunta:** ¿Qué sucede si NO escalamos variables con diferentes unidades?

Ejemplo:
```
ingreso:  [20000, 50000, 100000]  (varianza: 1.6e9)
edad:     [25, 35, 45]            (varianza: 100)
```

**Sin escalar:**
- PC1 ≈ 0.999·ingreso + 0.001·edad
- Ingreso domina por varianza mayor

**Con escalar:**
- PC1 ≈ 0.7·ingreso + 0.7·edad
- Ambas variables contribuyen equitativamente

### Ejercicio 5: Aplicación

**Tarea:** Diseña un experimento PCA para:

Dataset: Información de casas
```
Variables: precio, área, habitaciones, baños, antigüedad, distancia_centro
```

**Pasos:**
1. ¿Escalarías las variables? ¿Por qué?
2. ¿Cuántos componentes esperarías retener?
3. ¿Cómo interpretarías PC1 y PC2?
4. ¿Usarías PCA o Sparse PCA? Justifica.

**Respuesta sugerida:**
1. **Sí escalar**: diferentes unidades (m², años, km, $)
2. **2-3 componentes**: variables probablemente correlacionadas
3. Interpretación posible:
   - PC1: "Tamaño/calidad general" (área, habitaciones, baños, precio)
   - PC2: "Ubicación vs antigüedad" (distancia vs edad)
4. **PCA regular**: pocas variables, todas importantes (usar Sparse si hubiera 50+ variables)

---

## Preguntas Frecuentes

### 1. ¿PCA es supervisado o no supervisado?

**No supervisado**: No usa etiquetas/clases. Solo busca estructura en X.

### 2. ¿PCA elimina variables o las combina?

**Combina**: Crea nuevas variables (PCs) como combinaciones lineales de originales.

### 3. ¿Se puede recuperar la información original?

**Sí, parcialmente**: 
```python
X_aprox = scores · loadingsᵀ
```
Si usamos todas las PCs, recuperación perfecta. Si usamos k < p, aproximación.

### 4. ¿PCA asume distribución normal?

**No necesariamente**: Funciona sin asumir normalidad, pero funciona mejor con datos aproximadamente normales.

### 5. ¿Qué hacer con valores faltantes?

**Opciones:**
1. Eliminar filas con NAs (como en código: `dropna()`)
2. Imputar valores (media, mediana, KNN)
3. Usar variantes de PCA robustas a missing data

### 6. ¿PCA maneja outliers bien?

**No muy bien**: Outliers pueden distorsionar los componentes. 
**Solución**: Remover outliers o usar Robust PCA.

---

## Resumen de Conceptos Clave

### Matemáticos
- ✅ PCA descompone matriz de covarianza en valores/vectores propios
- ✅ Componentes principales = direcciones de máxima varianza
- ✅ Componentes son ortogonales (no correlacionados)
- ✅ Valores propios ordenados: λ₁ ≥ λ₂ ≥ ... ≥ λₚ

### Prácticos
- ✅ Escalar datos si diferentes unidades
- ✅ Retener componentes que expliquen 80-90% varianza
- ✅ Primeras 2-3 PCs usualmente suficientes
- ✅ Visualizar con scree plot y biplot

### Interpretación
- ✅ Loadings: cómo se forma cada PC
- ✅ Scores: posición de observaciones en nuevo espacio
- ✅ Varianza explicada: importancia de cada PC
- ✅ Sparse PCA: mejor interpretabilidad

### Limitaciones
- ⚠️ Solo para variables continuas
- ⚠️ Asume relaciones lineales
- ⚠️ Sensible a outliers
- ⚠️ Interpretación puede ser difícil

---

## Referencias para Profundizar

### Libros
1. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - Capítulo 14: PCA y reducción de dimensionalidad

2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - Capítulo 12: PCA y análisis factorial

3. **"Applied Multivariate Statistical Analysis"** - Johnson & Wichern
   - Capítulo 8: PCA

### Artículos
- Jolliffe, I.T. (2002). "Principal Component Analysis"
- Zou, Hastie & Tibshirani (2006). "Sparse Principal Component Analysis"

### Recursos Online
- StatQuest: Videos explicativos de PCA
- Scikit-learn documentation: Ejemplos prácticos

---

## Checklist de Estudio

Marca cuando domines cada concepto:

### Conceptos Básicos
- [ ] Definición de PCA
- [ ] Valores y vectores propios
- [ ] Matriz de covarianza
- [ ] Diferencia entre loadings y scores

### Aplicación
- [ ] Cuándo usar PCA
- [ ] Cuándo NO usar PCA
- [ ] Necesidad de escalar datos
- [ ] Interpretar resultados de PCA

### Avanzado
- [ ] Selección de número de componentes
- [ ] Sparse PCA
- [ ] Variantes de PCA
- [ ] Limitaciones y alternativas

---

**¡Buena suerte en tu estudio! 📊🎓**

