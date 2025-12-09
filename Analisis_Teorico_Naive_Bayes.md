# Análisis Teórico: Clasificación con Naive Bayes para Aceptación de Préstamos Personales

## Índice
1. [Introducción y Contexto](#introducción-y-contexto)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
4. [Partición de Datos](#partición-de-datos)
5. [Análisis de Tablas de Contingencia](#análisis-de-tablas-de-contingencia)
6. [Teoría de Naive Bayes](#teoría-de-naive-bayes)
7. [Implementación Práctica](#implementación-práctica)
8. [Evaluación del Modelo](#evaluación-del-modelo)
9. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## Introducción y Contexto

### Problema de Negocio
El Universal Bank necesita predecir qué clientes aceptarán ofertas de préstamos personales para optimizar sus campañas de marketing y reducir costos operativos. Con solo un 9.6% de tasa de aceptación (480 de 5000 clientes), es crucial identificar patrones predictivos.

### Variables de Estudio
- **Variable Objetivo (Target)**: `Personal Loan` (binaria: 0=No acepta, 1=Acepta)
- **Variables Predictoras**:
  - `Online`: Uso de servicios bancarios en línea (binaria: 0=No, 1=Sí)
  - `CreditCard`: Tenencia de tarjeta de crédito del banco (binaria: 0=No, 1=Sí)

### Justificación de la Selección
Estas variables fueron elegidas porque:
1. **Online**: Indica nivel de digitalización y comodidad del cliente
2. **CreditCard**: Muestra relación crediticia previa con el banco
3. **Ambas son binarias**: Simplifican el análisis y permiten interpretación clara

---

## Fundamentos Teóricos

### 1. Teorema de Bayes

**Fórmula Fundamental:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Donde:**
- `P(A|B)`: Probabilidad posterior
- `P(B|A)`: Verosimilitud (likelihood)
- `P(A)`: Probabilidad previa (prior)
- `P(B)`: Evidencia (marginal likelihood)

### 2. Clasificador Naive Bayes

**Suposición de Independencia Condicional:**
```
P(X₁, X₂, ..., Xₙ | Y) = ∏ᵢ₌₁ⁿ P(Xᵢ | Y)
```

**Fórmula de Clasificación:**
```
P(Y = y | X₁, X₂, ..., Xₙ) = P(Y = y) × ∏ᵢ₌₁ⁿ P(Xᵢ | Y = y) / P(X₁, X₂, ..., Xₙ)
```

**Para nuestro caso específico:**
```
P(Loan = 1 | CC = 1, Online = 1) = P(Loan = 1) × P(CC = 1 | Loan = 1) × P(Online = 1 | Loan = 1) / P(CC = 1, Online = 1)
```

### 3. Ley de Probabilidad Total

**Fórmula:**
```
P(A) = ∑ᵢ P(A | Bᵢ) × P(Bᵢ)
```

**Aplicada a nuestro problema:**
```
P(CC = 1, Online = 1) = P(CC = 1, Online = 1 | Loan = 1) × P(Loan = 1) + P(CC = 1, Online = 1 | Loan = 0) × P(Loan = 0)
```

---

## Análisis Exploratorio de Datos

### 1. Estadísticas Descriptivas

**Distribución de la Variable Objetivo:**
- **Clase Mayoritaria (No acepta)**: 4520 casos (90.4%)
- **Clase Minoritaria (Acepta)**: 480 casos (9.6%)

**Implicaciones:**
- **Desequilibrio de clases**: Requiere técnicas especiales de evaluación
- **Tasa base**: 9.6% - cualquier modelo debe superar esta precisión

### 2. Análisis de Variables Predictoras

**Variable Online:**
- Media: 0.597 (59.7% usa servicios en línea)
- Distribución: Relativamente balanceada

**Variable CreditCard:**
- Media: 0.294 (29.4% tiene tarjeta de crédito)
- Distribución: Sesgada hacia "No tiene"

### 3. Técnicas de Análisis Exploratorio

**Métodos Utilizados:**
1. **Estadísticas descriptivas**: Media, desviación estándar, cuartiles
2. **Distribuciones de frecuencia**: Conteos y proporciones
3. **Análisis de correlación**: Relaciones entre variables

---

## Partición de Datos

### 1. Estrategia de División

**División 60-40:**
- **Entrenamiento**: 3000 casos (60%)
- **Validación**: 2000 casos (40%)

### 2. Técnica: Train-Test Split

**Parámetros:**
```python
train_test_split(X, y, test_size=0.40, random_state=1)
```

**Justificación del random_state=1:**
- **Reproducibilidad**: Garantiza resultados consistentes
- **Comparabilidad**: Permite comparar diferentes modelos
- **Debugging**: Facilita la identificación de problemas

### 3. Estratificación Implícita

**Verificación de Proporciones:**
- **Entrenamiento**: 9.57% acepta préstamos
- **Validación**: 9.65% acepta préstamos
- **Diferencia**: 0.08% (aceptable para muestras grandes)

**Importancia:**
- Mantiene la distribución de clases en ambos conjuntos
- Evita sesgo en el entrenamiento

---

## Análisis de Tablas de Contingencia

### 1. Tabla Pivot Principal

**Estructura:**
```
                    Online=0    Online=1
CC    Loan
0     0              792        1117
      1               73         126
1     0              327         477
      1               39          49
```

**Interpretación:**
- **Fila (CC=1, Loan=1, Online=1)**: 49 casos
- **Total (CC=1, Online=1)**: 49 + 477 = 526 casos
- **Probabilidad condicional**: 49/526 = 0.0932

### 2. Técnicas de Análisis

**Métodos Utilizados:**
1. **Groupby y Pivot**: Agregación y reorganización de datos
2. **Melt y Pivot**: Transformación de estructura de datos
3. **Cálculo de probabilidades condicionales**

### 3. Tablas Pivot Separadas

**Tabla Loan vs Online:**
```
Online     0     1
Loan              
0       1119  1594
1        112   175
```

**Tabla Loan vs CreditCard:**
```
CC       0    1
Loan          
0     1909  804
1      199   88
```

**Propósito:**
- Simplificar cálculos de probabilidades marginales
- Facilitar la interpretación de relaciones bivariadas

---

## Teoría de Naive Bayes

### 1. Cálculo Manual de Probabilidades

**Probabilidades Requeridas:**

1. **P(CC = 1 | Loan = 1) = 88/287 = 0.3066**
   - 30.66% de los que aceptan préstamos tienen tarjeta de crédito

2. **P(Online = 1 | Loan = 1) = 175/287 = 0.6098**
   - 60.98% de los que aceptan préstamos usan servicios en línea

3. **P(Loan = 1) = 287/3000 = 0.0957**
   - 9.57% de la población acepta préstamos

4. **P(CC = 1 | Loan = 0) = 804/2713 = 0.2964**
   - 29.64% de los que no aceptan tienen tarjeta de crédito

5. **P(Online = 1 | Loan = 0) = 1594/2713 = 0.5875**
   - 58.75% de los que no aceptan usan servicios en línea

6. **P(Loan = 0) = 2713/3000 = 0.9043**
   - 90.43% de la población no acepta préstamos

### 2. Aplicación de la Fórmula de Bayes

**Cálculo de P(CC = 1, Online = 1):**
```
P(CC = 1, Online = 1) = P(CC = 1 | Loan = 1) × P(Online = 1 | Loan = 1) × P(Loan = 1) + 
                        P(CC = 1 | Loan = 0) × P(Online = 1 | Loan = 0) × P(Loan = 0)
                      = 0.3066 × 0.6098 × 0.0957 + 0.2964 × 0.5875 × 0.9043
                      = 0.0179 + 0.1574
                      = 0.1753
```

**Cálculo de P(Loan = 1 | CC = 1, Online = 1):**
```
P(Loan = 1 | CC = 1, Online = 1) = (0.3066 × 0.6098 × 0.0957) / 0.1753
                                 = 0.0179 / 0.1753
                                 = 0.1020
```

### 3. Comparación de Métodos

**Resultados:**
- **Tabla Pivot (Directa)**: 0.0932
- **Naive Bayes (Manual)**: 0.1020
- **Diferencia**: 0.0088 (8.8% relativo)

**Interpretación:**
- La diferencia se debe a la **suposición de independencia condicional**
- Naive Bayes asume que CC y Online son independientes dado Loan
- En realidad, pueden existir correlaciones no capturadas

---

## Implementación Práctica

### 1. Multinomial Naive Bayes

**Clase Utilizada:**
```python
from sklearn.naive_bayes import MultinomialNB
```

**Parámetros:**
- **alpha = 0.01**: Suavizado de Laplace (evita probabilidades cero)
- **fit_prior = True**: Aprende probabilidades previas de los datos

### 2. Suavizado de Laplace

**Fórmula:**
```
P(Xᵢ = x | Y = y) = (count(Xᵢ = x, Y = y) + α) / (count(Y = y) + α × n_features)
```

**Donde:**
- α = 0.01 (parámetro de suavizado)
- n_features = 2 (número de características)

**Propósito:**
- Evita probabilidades de cero
- Mejora la generalización
- Maneja casos no vistos en entrenamiento

### 3. Resultados del Modelo

**Probabilidad Predicha:**
- **Scikit-learn**: 0.0956
- **Muy cercana a la tabla pivot**: 0.0932
- **Diferencia**: 0.0024 (2.4% relativo)

**Interpretación:**
- El modelo está bien calibrado
- Las probabilidades son realistas
- El suavizado funciona correctamente

---

## Evaluación del Modelo

### 1. Matriz de Confusión

**Resultados:**
```
       Prediction
Actual    0    1
     0 1807    0
     1  193    0
```

**Métricas Calculadas:**
- **Precisión (Accuracy)**: 1807/2000 = 0.9035 (90.35%)
- **Sensibilidad (Recall)**: 0/193 = 0.0000 (0%)
- **Especificidad**: 1807/1807 = 1.0000 (100%)
- **Precisión (Precision)**: Indefinida (0 predicciones positivas)

### 2. Análisis de Rendimiento

**Problemas Identificados:**
1. **Sesgo hacia la clase mayoritaria**: El modelo predice siempre "No acepta"
2. **Falta de sensibilidad**: No detecta casos positivos
3. **Umbral de decisión**: Probablemente muy alto (0.5)

### 3. Técnicas de Mejora

**Recomendaciones:**
1. **Ajuste de umbral**: Reducir el umbral de decisión
2. **Balanceo de clases**: SMOTE, undersampling, o pesos de clase
3. **Métricas alternativas**: F1-score, AUC-ROC, precisión balanceada
4. **Validación cruzada**: K-fold para evaluación más robusta

---

## Conclusiones y Recomendaciones

### 1. Hallazgos Principales

**Probabilidades Calculadas:**
- **P(Loan = 1 | CC = 1, Online = 1) = 0.0932** (desde tabla pivot)
- **Diferencia entre métodos**: 0.0088 (aceptable para fines prácticos)

**Interpretación de Negocio:**
- Clientes con tarjeta de crédito y uso de servicios en línea tienen **9.32% de probabilidad** de aceptar préstamos
- Esta probabilidad es **ligeramente menor** que la tasa base (9.6%)
- **No hay evidencia fuerte** de que estas características aumenten la probabilidad de aceptación

### 2. Limitaciones del Análisis

**Técnicas:**
1. **Suposición de independencia**: CC y Online pueden estar correlacionados
2. **Desequilibrio de clases**: Afecta la evaluación del modelo
3. **Variables limitadas**: Solo 2 predictores considerados
4. **Modelo simple**: Multinomial NB puede ser demasiado básico

### 3. Recomendaciones Técnicas

**Mejoras del Modelo:**
1. **Incluir más variables**: Income, Age, Education, etc.
2. **Técnicas de balanceo**: SMOTE, ADASYN, o pesos de clase
3. **Modelos alternativos**: Random Forest, XGBoost, SVM
4. **Validación cruzada**: K-fold estratificado
5. **Optimización de hiperparámetros**: Grid search o Bayesian optimization

**Métricas de Evaluación:**
1. **AUC-ROC**: Mejor para clases desbalanceadas
2. **F1-Score**: Balance entre precisión y recall
3. **Precision-Recall Curve**: Específica para clases minoritarias
4. **Matriz de confusión normalizada**: Para comparar modelos

### 4. Aplicación Práctica

**Estrategia de Marketing:**
1. **Segmentación**: Identificar subgrupos con mayor probabilidad
2. **Personalización**: Adaptar mensajes según características
3. **Optimización de costos**: Enfocar recursos en clientes prometedores
4. **Monitoreo continuo**: Actualizar modelo con nuevos datos

### 5. Fórmulas Matemáticas Resumidas

**Teorema de Bayes:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Naive Bayes:**
```
P(Y|X₁,X₂,...,Xₙ) = P(Y) × ∏ᵢ P(Xᵢ|Y) / P(X₁,X₂,...,Xₙ)
```

**Ley de Probabilidad Total:**
```
P(A) = ∑ᵢ P(A|Bᵢ) × P(Bᵢ)
```

**Suavizado de Laplace:**
```
P(Xᵢ|Y) = (count(Xᵢ,Y) + α) / (count(Y) + α × n_features)
```

---

## Referencias Técnicas

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. The Elements of Statistical Learning
2. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013)**. An Introduction to Statistical Learning
3. **Murphy, K. P. (2012)**. Machine Learning: A Probabilistic Perspective
4. **Bishop, C. M. (2006)**. Pattern Recognition and Machine Learning
5. **Scikit-learn Documentation**: Naive Bayes Classifiers

---

*Este documento proporciona una base teórica sólida para entender y aplicar el análisis de Naive Bayes en problemas de clasificación binaria con clases desbalanceadas.*





