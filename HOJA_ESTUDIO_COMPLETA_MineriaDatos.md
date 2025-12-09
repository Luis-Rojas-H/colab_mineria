# HOJA DE ESTUDIO COMPLETA: MINERÍA DE DATOS (8vo Ciclo)
## Multinomial Naive Bayes y Árboles de Decisión

**Autor:** Minería de Datos - Ciclo VIII  
**Temas:** Clasificación Bayesiana y Métodos basados en Árboles  
**Fecha:** 2025

---

# PARTE 1: FUNDAMENTOS DE CLASIFICACIÓN

## 1.1 Conceptos Básicos

### ¿Qué es Clasificación?
La clasificación es una tarea de aprendizaje supervisado donde:
- **Objetivo:** Predecir una clase o categoría para nuevas observaciones
- **Entrada (X):** Variables predictoras (características)
- **Salida (y):** Variable categórica (clases)

**Matemáticamente:**
```
Dado: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} conjunto de entrenamiento

Encontrar: h(x) = función de hipótesis tal que:
h(xᵢ) ≈ yᵢ para datos nuevos
```

### Evaluación de Clasificadores

**Matriz de Confusión:**
```
                    Predicción
                Positive  Negative
Real Positive      TP        FN
Real Negative      FP        TN
```

**Métricas:**
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)  ← ¿De mis predicciones positivas, cuántas eran correctas?
- **Recall** = TP / (TP + FN)     ← ¿Cuántos positivos reales detecté?
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

# PARTE 2: MULTINOMIAL NAIVE BAYES

## 2.1 Teoría Fundamental

### ¿Qué es Naive Bayes?

**Definición:** Clasificador probabilístico basado en el Teorema de Bayes que asume independencia condicional entre predictores.

**El nombre "Naive" (Ingenuo):** Porque asume que los predictores son independientes entre sí, cuando en realidad frecuentemente no lo son.

### 2.2 Desarrollo Matemático del Teorema de Bayes

**Teorema de Bayes (Forma Básica):**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**En contexto de clasificación:**
```
P(Clase | Datos) = P(Datos | Clase) × P(Clase) / P(Datos)
```

**Donde:**
- P(Clase | Datos): Probabilidad posterior (lo que queremos calcular)
- P(Datos | Clase): Verosimilitud (likelihood)
- P(Clase): Probabilidad prior (probabilidad base de la clase)
- P(Datos): Evidencia (normalización)

### 2.3 Extensión a Múltiples Características

**Para múltiples predictores:**
```
P(y | x₁, x₂, ..., xₙ) = P(x₁, x₂, ..., xₙ | y) × P(y) / P(x₁, x₂, ..., xₙ)
```

**Asumiendo independencia condicional:**
```
P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

**Por lo tanto:**
```
P(y | x₁, x₂, ..., xₙ) ∝ P(y) × ∏ᵢ P(xᵢ | y)
```

**Predicción:**
```
ŷ = argmax_y [log(P(y)) + Σᵢ log(P(xᵢ | y))]
```

### 2.4 Caso: Multinomial Naive Bayes

**Multinomial Naive Bayes** es específicamente para:
- Variables **discretas** o **conteos**
- Datos que pueden representarse como vectores de conteos

**Estimación de probabilidades:**

Para una característica j y clase y:
```
P(xⱼ | y) = (conteos de xⱼ en clase y + α) / (total de conteos en clase y + α × V)
```

Donde:
- α = parámetro de Laplace smoothing (típicamente 1)
- V = número de valores únicos de la característica j

**¿Por qué Laplace Smoothing?**
- Evita probabilidades cero
- Si un valor nunca aparece en training, no dice P=0
- Permite generalización a casos nuevos

### 2.5 Implementación en el Notebook

```python
delays_nb = MultinomialNB(alpha=0.01)
delays_nb.fit(X_train, y_train)
```

**Pasos ejecutados internamente:**
1. Calcula P(Clase) de cada clase en training
2. Para cada característica, calcula P(característica | Clase)
3. Aplica Laplace smoothing (alpha=0.01)
4. Almacena estos parámetros

**Predicción:**
```python
y_pred = delays_nb.predict(X_test)
probabilities = delays_nb.predict_proba(X_test)
```

---

## 2.6 ¿Por qué se usan VARIABLES DISCRETAS en Naive Bayes?

### Razones Teóricas

**1. Estimación de Probabilidades:**
```
Variables DISCRETAS:
├─ Fácil contar ocurrencias
├─ P(x | y) = (conteo) / (total)
└─ Estimación directa y confiable

Variables CONTINUAS:
├─ Infinitas posibilidades
├─ Necesita asumir distribución (ej: Gaussiana)
└─ Más parámetros a estimar
```

**2. Multinomial específicamente:**
- Asume que cada característica es un recuento
- Funciona con frecuencias
- Ideal para clasificación de texto (bag of words)

**3. En el notebook FlightDelays:**
- WEATHER_R: 0, 1, 2 (categorías)
- DAY_WEEK: 1, 2, 3, 4, 5, 6, 7 (categorías)
- CARRIER: AA, DL, OH, etc. (categorías)

**Todas convertidas a variables binarias (one-hot encoding):**
```
CARRIER_AA = [0, 1, 0, 0, ...]
CARRIER_DL = [1, 0, 0, 0, ...]
CARRIER_OH = [0, 0, 1, 0, ...]
```

### Comparación Teórica

```
┌─────────────────────┬──────────────────┬──────────────────────┐
│ Característica      │ Discretas        │ Continuas            │
├─────────────────────┼──────────────────┼──────────────────────┤
│ P(x|y) estimación   │ Directo (conteo) │ Asume distribución   │
│ Parámetros          │ Pocos            │ Muchos               │
│ Interpretabilidad   │ Alta             │ Media                │
│ Overfitting         │ Bajo riesgo      │ Mayor riesgo         │
│ Escalabilidad       │ Excelente        │ Buena                │
│ Uso en texto        │ Ideal            │ No recomendado       │
└─────────────────────┴──────────────────┴──────────────────────┘
```

### Transformación de Continuas a Discretas

Si tienes datos continuos pero quieres usar Multinomial NB, debes:

```python
# Opción 1: Binning (crear intervalos)
df['edad_binned'] = pd.cut(df['edad'], bins=[0, 18, 30, 50, 100])

# Opción 2: Discretización por cuantiles
df['ingresos_categoria'] = pd.qcut(df['ingresos'], q=4)

# Opción 3: Usar Gaussian Naive Bayes (asume distribución normal)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()  # Para variables continuas
```

---

# PARTE 3: ÁRBOLES DE DECISIÓN (CART)

## 3.1 Conceptos Fundamentales

### ¿Qué es un Árbol de Decisión?

**Definición:** Modelo de clasificación que divide recursivamente el espacio de características en regiones puras (homogéneas en términos de clase).

**Estructura:**
```
              Nodo Raíz
             (Decisión)
            /          \
        Sí/            No\
       /                  \
    Nodo Interno         Nodo Interno
    /      \            /        \
  Hoja    Hoja       Hoja       Hoja
(Clase A)(Clase B)  (Clase C)  (Clase A)
```

### 3.2 Desarrollo Matemático

#### Impureza de Gini

**Definición:** Mide la probabilidad de clasificar erróneamente un elemento tomado aleatoriamente.

```
Gini(S) = 1 - Σⱼ pⱼ²

Donde:
- S = conjunto de muestras
- pⱼ = proporción de clase j en S
```

**Ejemplo:**
```
Si tenemos 100 muestras:
- 60 de Clase A
- 40 de Clase B

Gini = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48
```

**Interpretación:**
- Gini = 0 → Nodo puro (una sola clase)
- Gini = 0.5 → Máxima impureza (50-50)
- Gini > 0.4 → Nodo muy impuro

#### Ganancia de Información (Information Gain)

**Definición:** Reducción en impureza después de dividir por una característica.

```
IG(S, A) = Gini(S) - Σᵥ (|Sᵥ| / |S|) × Gini(Sᵥ)

Donde:
- S = conjunto original
- A = característica de división
- Sᵥ = subconjuntos después de división
- |Sᵥ| / |S| = proporción ponderada
```

**Algoritmo de División:**
```
Para cada característica A:
    Para cada valor v en A:
        Calcular IG(S, A)
    
Elegir división con máximo IG
```

### 3.3 Algoritmo de Construcción (ID3/C4.5/CART)

```
función ConstructorArbol(S, características):
    1. Si S es pura (una sola clase):
        Retornar hoja con esa clase
    
    2. Si no hay más características:
        Retornar hoja con clase mayoritaria
    
    3. Para cada característica:
        Calcular ganancia de información
    
    4. Seleccionar característica con máxima ganancia
    
    5. Para cada valor de esa característica:
        Crear rama
        Recursivamente construir árbol con subset
    
    6. Retornar nodo con ramas
```

### 3.4 Parámetros Clave de Control

```python
DecisionTreeClassifier(
    max_depth=10,                    # Profundidad máxima
    min_samples_split=20,            # Mínimas muestras para dividir
    min_samples_leaf=10,             # Mínimas muestras por hoja
    min_impurity_decrease=0.01,      # Mejora mínima de impureza
    criterion='gini'                 # Métrica ('gini' o 'entropy')
)
```

**¿Por qué estos parámetros?**
```
max_depth → Evita árboles demasiado profundos (overfitting)
min_samples_split → Evita divisiones sobre pocas muestras
min_impurity_decrease → Solo divide si mejora significativamente
```

### 3.5 Problema de Overfitting en Árboles

**Árbol Completo vs Podado:**
```
Árbol Sin Restricciones:
├─ Accuracy Training: 100%  ← Memoriza datos
├─ Accuracy Validation: 75% ← No generaliza
└─ Problema: OVERFITTING

Árbol Con Restricciones:
├─ Accuracy Training: 85%  ← Más simple
├─ Accuracy Validation: 84%← Generaliza bien
└─ Solución: REGULARIZACIÓN
```

**En el notebook:**
```python
# Árbol sin restricciones
fullClassTree = DecisionTreeClassifier(random_state=1)
# Training Accuracy: 1.0000, Validation: 0.9790 ← Overfitting

# Árbol podado
smallClassTree = DecisionTreeClassifier(
    max_depth=30, 
    min_samples_split=20,
    min_impurity_decrease=0.01
)
# Training: 0.9823, Validation: 0.9770 ← Mejor balance
```

---

# PARTE 4: RANDOM FOREST (Ensambles)

## 4.1 ¿Qué es Random Forest?

**Definición:** Conjunto (ensamble) de árboles de decisión entrenados independientemente con muestras bootstrap aleatorias.

**Ventajas:**
```
Árbol Individual:
├─ Interpretable
├─ Rápido
└─ Prone a overfitting

Random Forest:
├─ Muy robusto (reduce overfitting)
├─ Mejor generalización
├─ Proporciona Feature Importance
└─ Desventaja: Menos interpretable
```

## 4.2 Matemática de Random Forest

### Bootstrap Aggregating (Bagging)

**Proceso:**
```
Para i = 1 a B (número de árboles):
    1. Tomar muestra bootstrap: Sᵢ ~ muestreo con reemplazo de S
    2. Entrenar árbol Tᵢ con Sᵢ
    3. Guardar predicción

Predicción final:
├─ Clasificación: modo (votación mayoritaria)
├─ Regresión: promedio
```

### Feature Importance en Random Forest

**Cálculo:**
```
Para cada característica j:
    1. En cada árbol i, calcular ganancia_ij
    2. Importancia_j = Promedio(ganancia_ij para todos i)
    3. Normalizar al rango [0, 1]
```

**Interpretación:**
```
Característica A: 0.35 ← Muy importante (35% de importancia)
Característica B: 0.25 ← Importante
Característica C: 0.05 ← Poco importante
```

**En el notebook:**
```
Feature Importance (UniversalBank):
Income:     0.3338 ← La más importante
Education:  0.2008
CCAvg:      0.1721
... (continúa)
```

---

# PARTE 5: SELECCIÓN DE CARACTERÍSTICAS

## 5.1 ¿Por qué seleccionar características?

**Razones:**
```
1. Curse of Dimensionality
   └─ Más características = más datos necesarios

2. Reducción de Ruido
   └─ No todas las características son informativas

3. Interpretabilidad
   └─ Modelos más simples y explicables

4. Eficiencia
   └─ Menos características = menos computación
```

## 5.2 Métodos de Selección

### Método 1: Filter (Independiente del Modelo)

```
Pasos:
1. Calcular correlación con variable objetivo
2. Seleccionar top-k características con mayor correlación
3. Entrenar modelo
```

**Ventaja:** Rápido  
**Desventaja:** Ignora interacciones entre características

### Método 2: Wrapper (Dependiente del Modelo)

```
Pasos:
1. Comenzar con conjunto S de características
2. Eliminar feature con menor contribución
3. Entrenar modelo y evaluar
4. Si mejora, mantener
5. Repetir hasta convergencia
```

**Ventaja:** Considera interacciones  
**Desventaja:** Lento y costoso

### Método 3: Embedded (Dentro del Modelo)

```
Árboles de Decisión:
└─ Feature Importance

Random Forest:
└─ Feature Importance (promedio de árboles)

Regularización (L1):
└─ Coeficientes pequeños = features menos importantes
```

## 5.3 En Nuestros Notebooks

### Multinomial Naive Bayes (FlightDelays)

```python
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST', 'CARRIER']
```

**¿Por qué estas columnas?**
1. Todas son categóricas/discretas
2. Tienen relación lógica con Flight Status
3. Son fáciles de obtener (información pre-vuelo)
4. No requieren transformación compleja

**¿Por qué NOT otras?**
- Algunas no estaban disponibles en tiempo de predicción
- Algunas eran redundantes
- Algunas tenían muy poco poder predictivo

### Decision Trees/Random Forest (UniversalBank)

```python
X = bank_df.drop(columns=['Personal Loan', 'ID', 'ZIP Code'])
```

**¿Por qué excluir ID y ZIP Code?**
```
ID: 
└─ Identificador, no característica

ZIP Code:
└─ Demasiadas categorías
└─ No ordenadas geográficamente en el modelo
```

**Todas las columnas restantes:**
- Income (continua) → Correlacionada con capacidad de pago
- Age (continua) → Predictor de riesgo
- Family (discreta) → Tamaño familiar → Necesidades de crédito
- Education (discreta) → Proxy de ingresos futuros
- Experience (continua) → Relacionado con estabilidad
- etc.

---

# PARTE 6: VARIABLES CONTINUAS vs DISCRETAS

## 6.1 Tabla Comparativa Completa

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ Aspecto          │ CONTINUAS            │ DISCRETAS            │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Ejemplos         │ Edad, Ingresos       │ Día_semana, Genero   │
│ Rango de valores │ Infinitos en rango   │ Finitos y específicos│
│ Distribución     │ Gaussiana, Uniform   │ Categórica           │
│ Representación   │ ℝ (números reales)   │ ℤ o Categorías       │
├──────────────────┼──────────────────────┼──────────────────────┤
│ NAIVE BAYES      │                      │                      │
│ ├─ MultinomialNB │ ❌ No recomendado    │ ✅ Ideal             │
│ ├─ GaussianNB    │ ✅ Recomendado       │ ❌ No aplica         │
│ └─ Bernoulli     │ ❌ No recomendado    │ ✅ Binarias solo     │
├──────────────────┼──────────────────────┼──────────────────────┤
│ ÁRBOLES          │ ✅ Excelente         │ ✅ Excelente         │
│ ├─ Decision Tree │ ✅ Maneja bien       │ ✅ Maneja bien       │
│ └─ Random Forest │ ✅ Maneja bien       │ ✅ Maneja bien       │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Transformación   │ Normalizar/Escalar   │ One-hot encoding     │
│ requerida        │ o Binning            │ o Label encoding     │
└──────────────────┴──────────────────────┴──────────────────────┘
```

## 6.2 ¿Por qué Naive Bayes necesita variables discretas?

**Razón matemática:**

```
MultinomialNB estima:
P(x_j | y) = (conteos de x_j en clase y) / (total de conteos)

Si x_j es continua (ej: edad=34.567):
└─ Prácticamente no se repite exactamente
└─ Conteos = 0 o 1 principalmente
└─ Estimación poco confiable
```

**Solución:**
```
Convertir continua a discreta:
Age: 34.567 → Edad_Bin_30_40

Ahora:
P(Edad_30_40 | Pide_Prestamo) 
= (cuántos pidieron préstamo entre 30-40) / (total que piden préstamo)
= Estimable y confiable
```

## 6.3 Transformaciones Clave

### Para Multinomial Naive Bayes

```python
# Opción 1: Binning equiancho
df['edad_categoria'] = pd.cut(df['edad'], bins=[0, 30, 40, 50, 100])

# Opción 2: Binning equifrecuencia
df['edad_categoria'] = pd.qcut(df['edad'], q=4)

# Opción 3: One-hot encoding (como en nuestro notebook)
X = pd.get_dummies(df[['DAY_WEEK', 'CARRIER', 'ORIGIN']])
# Resultado: variables binarias 0/1
```

### Para Decision Trees

```python
# NO es necesario transformar
# Los árboles manejan tanto continuas como discretas
tree = DecisionTreeClassifier()
tree.fit(X_con_continuas_y_discretas, y)
```

---

# PARTE 7: SELECCIÓN PRIMERAS COLUMNAS CON MAYOR INFORMACIÓN

## 7.1 ¿Qué significa "Mayor Información"?

**En minería de datos, "información" se mide como:**

```
1. GANANCIA DE INFORMACIÓN (Entropy/Gini)
   └─ Cuánto reduce la impureza en el árbol

2. CORRELACIÓN
   └─ Fuerza de relación lineal con objetivo

3. MUTUAL INFORMATION
   └─ Información compartida entre X e y

4. CHI-CUADRADO (para variables categóricas)
   └─ Dependencia estadística
```

## 7.2 Métodos Prácticos de Selección

### Método 1: Correlation Matrix

```python
import pandas as pd
import numpy as np

# Calcular correlación con variable objetivo
correlations = df.corr()['target'].sort_values(ascending=False)

# Seleccionar top-k
top_features = correlations.index[1:6]  # Top 5 (excluir target)
```

**Interpretación:**
```
Income    vs Personal_Loan: 0.45 ← Fuerte positivo
Age       vs Personal_Loan: 0.32 ← Moderado positivo
Experience vs Personal_Loan: 0.30 ← Moderado positivo
```

### Método 2: Feature Importance (Trees)

```python
# Entrenar árbol
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Obtener importancia
importances = tree.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Top-5
top_5 = feature_importance.head(5)
```

### Método 3: Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif

# Calcular mutual information
mutual_info = mutual_info_classif(X, y, random_state=0)

# Ranking
ranking = pd.DataFrame({
    'feature': X.columns,
    'mutual_info': mutual_info
}).sort_values('mutual_info', ascending=False)
```

## 7.3 En Nuestros Notebooks

### FlightDelays (Multinomial NB)

```python
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST', 'CARRIER']
```

**Razón:**
Información disponible ANTES del vuelo:
- Día de la semana → Patrón de tráfico
- Hora de salida → Congestión
- Aeropuertos → Infraestructura
- Aerolínea → Historial de puntualidad

**Excluidas:**
- DEP_TIME (disponible solo después)
- Weather (podría usarse)
- TAIL_NUM (específica de avión, no generalizable)

### UniversalBank (Decision Trees)

En el output de Feature Importance vimos:
```
Income:     0.3338 ← Mayor información
Education:  0.2008
CCAvg:      0.1721
Family:     0.1114
Age:        0.0363
```

**Por qué Income es la más importante:**
- Correlación más fuerte con Personal Loan
- Mejor divide los datos en loan/no-loan
- Reduce más la impureza de Gini

---

# PARTE 8: ONE-HOT ENCODING vs LABEL ENCODING

## 8.1 ¿Por qué One-Hot Encoding en Multinomial NB?

```python
# Opción 1: Label Encoding (MALO para Multinomial NB)
CARRIER_AA = 0
CARRIER_DL = 1
CARRIER_OH = 2

# Problema: El modelo asume ORDEN (1 > 0, 2 > 1)
# Pero AA no es "menor" que DL

# Opción 2: One-Hot Encoding (BUENO para Multinomial NB)
CARRIER_AA = [1, 0, 0]
CARRIER_DL = [0, 1, 0]
CARRIER_OH = [0, 0, 1]

# Ventaja: No asume orden, cada categoría es independiente
```

**En código:**
```python
X = pd.get_dummies(delays_df[predictors])
# Automáticamente crea variables binarias para cada categoría
```

---

# PARTE 9: VALIDACIÓN CRUZADA (Cross-Validation)

## 9.1 ¿Por qué es importante?

```
Sin Cross-Validation:
├─ Un solo train/test split
├─ Resultado depende de cómo dividimos
└─ No es confiable

Con K-Fold Cross-Validation:
├─ K divisiones diferentes
├─ Promedio de resultados
└─ Más confiable y robusto
```

## 9.2 Matemática

```
5-Fold Cross-Validation:

Fold 1: Train [2,3,4,5] | Test [1]
Fold 2: Train [1,3,4,5] | Test [2]
Fold 3: Train [1,2,4,5] | Test [3]
Fold 4: Train [1,2,3,5] | Test [4]
Fold 5: Train [1,2,3,4] | Test [5]

Score Final = Media(Score₁, Score₂, Score₃, Score₄, Score₅)
```

**En el notebook:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree, X, y, cv=5)
# scores = [0.9883, 0.9733, 0.9933, 0.9817, 0.9933]
# Promedio: 0.9860
```

---

# PARTE 10: GRIDSEARCHCV (Búsqueda de Hiperparámetros)

## 10.1 ¿Qué es?

**Búsqueda exhaustiva sobre grid de parámetros:**

```python
param_grid = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [20, 40, 60, 80, 100],
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01],
}

# Combinaciones posibles: 4 × 5 × 5 = 100
```

**Proceso:**
```
Para cada combinación:
    1. Entrenar modelo con esos parámetros
    2. Evaluar con 5-fold CV
    3. Guardar score
    
Seleccionar combinación con mejor score
```

---

# PARTE 11: TABLA RESUMEN COMPARATIVA FINAL

```
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ Característica     │ Multinomial NB     │ Decision Tree      │ Random Forest      │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┤
│ Tipo de datos      │ Discretos ideal    │ Continuo/Discreto │ Continuo/Discreto │
│ Interpretabilidad  │ Media              │ Muy Alta           │ Media              │
│ Velocidad train    │ Muy rápido         │ Rápido             │ Lento (muchos árboles)
│ Velocidad pred     │ Rápida             │ Muy rápida         │ Lenta (votación)   │
│ Overfitting        │ Bajo riesgo        │ Alto riesgo        │ Muy bajo riesgo    │
│ Regularización     │ Laplace smoothing  │ max_depth, etc     │ Inherente          │
│ Feature Importance │ NO (sencillo)      │ SÍ                 │ SÍ (excelente)     │
│ Escalabilidad      │ Muy buena          │ Buena              │ Media              │
│ Manejo valores fal │ Bien               │ Bien               │ Bien               │
│ No linealidad      │ No maneja bien     │ Excelente          │ Excelente          │
│ Interacciones      │ No captura         │ Captura natural    │ Captura natural    │
│ Curse of Dimensio  │ Afectado           │ No afectado        │ No afectado        │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

---

# PARTE 12: PREGUNTAS DE EXAMEN TÍPICAS

## 12.1 Preguntas Teóricas

**P1: ¿Por qué Naive Bayes se llama "naive"?**
R: Porque asume que los predictores son independientes entre sí, cuando en realidad pueden estar correlacionados.

**P2: ¿Cuál es la diferencia entre Gini e Información Gain?**
R: Gini mide la impureza de un nodo (0=puro, 0.5=máximamente impuro). Información Gain es la reducción en Gini después de una división.

**P3: ¿Por qué One-Hot Encoding en Naive Bayes?**
R: Para evitar que el modelo asuma orden entre categorías. Cada categoría es una variable binaria independiente.

**P4: ¿Cuándo usar MultinomialNB vs GaussianNB?**
R: MultinomialNB para datos discretos/conteos. GaussianNB para datos continuos que siguen distribución normal.

## 12.2 Preguntas Prácticas

**P5: ¿Cómo elegir características?**
R: Usar correlación, mutual information, feature importance de árboles, o métodos wrapper con validación cruzada.

**P6: ¿Cómo evitar overfitting en árboles?**
R: Limitando profundidad (max_depth), mínimas muestras por nodo, mínima mejora de impureza, o usando Random Forest.

**P7: ¿Cuándo es mejor Random Forest que un solo árbol?**
R: Cuando queremos reducir varianza y mejorar generalización. Random Forest es más robusto pero menos interpretable.

---

# PARTE 13: FÓRMULAS CLAVE PARA MEMORIZAER

```
TEOREMA DE BAYES:
P(y|X) ∝ P(X|y) × P(y)

GINI:
Gini(S) = 1 - Σⱼ pⱼ²

INFORMATION GAIN:
IG(S,A) = Gini(S) - Σᵥ (|Sᵥ|/|S|) × Gini(Sᵥ)

LAPLACE SMOOTHING:
P(xⱼ|y) = (conteos + α) / (total + α×V)

ACCURACY:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

PRECISION:
Precision = TP / (TP + FP)

RECALL:
Recall = TP / (TP + FN)

F1-SCORE:
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# CONCLUSIÓN Y RECOMENDACIONES FINALES

## Cuándo usar cada algoritmo:

```
MULTINOMIAL NAIVE BAYES:
✓ Datos categóricos/discretos
✓ Modelos interpretables
✓ Datasets pequeños
✓ Clasificación de texto
Ejemplo: FlightDelays (días, aeropuertos, aerolíneas)

DECISION TREES:
✓ Interpretabilidad máxima
✓ Relaciones no-lineales
✓ Datos mixtos (continuos + discretos)
✓ Cuando necesitas explicar "por qué"
Desventaja: Sobreajuste

RANDOM FOREST:
✓ Máxima precisión
✓ Datasets grandes
✓ Problemas complejos
✓ No necesitas máxima interpretabilidad
✓ Cuando quieres mejor generalización
Ejemplo: UniversalBank (predicción de préstamos)
```

---

**Documento de Estudio Completo - Minería de Datos 8vo Ciclo**  
**Temas: Multinomial Naive Bayes, Árboles de Decisión, Random Forest**
