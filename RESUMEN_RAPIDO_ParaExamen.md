# RESUMEN RÁPIDO PARA EXAMEN - MINERÍA DE DATOS
## Repaso de 30 minutos antes del parcial

---

## 🎯 MAPA CONCEPTUAL GENERAL

```
┌─────────────────────────────────────────────────────────────┐
│                    CLASIFICACIÓN SUPERVISADA                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┬────────────────┬─────────────────┐  │
│  │ PROBABILÍSTICA   │  BASADO EN     │  ENSAMBLES      │  │
│  │                  │  ÁRBOLES       │                 │  │
│  ├──────────────────┼────────────────┼─────────────────┤  │
│  │ Naive Bayes      │ Decision Tree  │ Random Forest   │  │
│  │                  │ (CART)         │                 │  │
│  │ • Usa Bayes      │ • Usa Gini/IG  │ • Votación      │  │
│  │ • Rápido         │ • Interpretable│ • Robusto       │  │
│  │ • Discretos      │ • Overfitting  │ • Feature IM    │  │
│  └──────────────────┴────────────────┴─────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ NAIVE BAYES EN 60 SEGUNDOS

### La Fórmula Clave
```
P(Clase | Datos) ∝ P(Datos | Clase) × P(Clase)
```

### ¿Cuándo usarlo?
```
✅ USAR                    ❌ NO USAR
- Datos discretos          - Variables continuas
- Rápido necesario         - Datos muy correlacionados
- Dataset pequeño          - Necesitas explicar "por qué"
- Clasificación de texto   
```

### Multinomial vs Gaussian
```
MULTINOMIAL NAIVE BAYES:
├─ Para: Datos categóricos/conteos
├─ Ejemplo: Texto, días de semana
└─ En notebook: FlightDelays ✅

GAUSSIAN NAIVE BAYES:
├─ Para: Datos continuos
├─ Ejemplo: Edad, ingresos
└─ En notebook: ❌ No se usó
```

### Parámetro Importante: Alpha (Laplace Smoothing)
```
alpha = 0.01  (en nuestro notebook)

¿Por qué?
- Evita P(característica) = 0
- Permite generalizar a casos nuevos
- Mayor alpha = más suavizado
```

### Tabla de Probabilidades en FlightDelays
```
┌──────────────┬──────────────┬────────────────┐
│ CARRIER      │ P(Delay)     │ P(OnTime)      │
├──────────────┼──────────────┼────────────────┤
│ Delta (DL)   │ 0.0958       │ 0.2040         │
│ AA           │ 0.0575       │ 0.0349         │
│ Southwest    │ 0.0690       │ 0.2181         │
│ US Airways   │ Baja         │ Alta           │
└──────────────┴──────────────┴────────────────┘

Interpretación: US Airways tiene más retrasos (probabilidad alta)
```

---

## 2️⃣ ÁRBOLES DE DECISIÓN EN 60 SEGUNDOS

### Concepto
```
Árbol = Serie de preguntas SI/NO
└─ Cada pregunta: ¿Característica > Valor?
└─ Objetivo: Minimizar impureza
```

### Métrica Clave: GINI
```
Gini = 1 - Σ(proporciones²)

Ejemplos:
- Nodo con [100 A, 0 B]:    Gini = 1 - (1² + 0²) = 0 (puro)
- Nodo con [50 A, 50 B]:    Gini = 1 - (0.5² + 0.5²) = 0.5 (impuro)
- Nodo con [60 A, 40 B]:    Gini = 1 - (0.6² + 0.4²) = 0.48 (impuro)

Interpretación: Gini bajo = nodo bueno ✅
```

### Ganancia de Información (Information Gain)
```
IG = Gini(Padre) - Media(Gini(Hijos))

Proceso:
1. Calcular Gini del nodo actual
2. Dividir por cada característica
3. Calcular Gini de hijos
4. Elegir división con mayor IG
```

### Parámetros Críticos
```
┌────────────────────┬──────────────────────────────────┐
│ max_depth          │ Profundidad máxima del árbol     │
│ └─ Sin límite      │ Memoriza datos (overfitting)     │
│ └─ Con límite=10   │ Generaliza mejor                 │
├────────────────────┼──────────────────────────────────┤
│ min_samples_split  │ Mínimas muestras para dividir    │
│ └─ Baja (5)        │ Demasiadas divisiones (ruido)    │
│ └─ Alta (20)       │ Menos divisiones (underfitting)  │
├────────────────────┼──────────────────────────────────┤
│ min_impurity_...   │ Solo divide si mejora bastante   │
│ └─ 0.01            │ Evita divisiones inútiles        │
└────────────────────┴──────────────────────────────────┘
```

### El Problema del Overfitting
```
Árbol SIN restricciones:
├─ Training: 100% ✅
├─ Validation: 75% ❌ ← PROBLEMA
└─ Causa: Memoriza datos de training

Árbol CON restricciones:
├─ Training: 85%
├─ Validation: 84% ✅ ← EXCELENTE
└─ Causa: Generaliza mejor
```

---

## 3️⃣ RANDOM FOREST EN 60 SEGUNDOS

### Concepto
```
Random Forest = 500 árboles votando

Proceso:
1. Tomar 500 muestras aleatorias (bootstrap)
2. Entrenar 1 árbol por cada muestra
3. Para predecir: votación mayoritaria
4. Resultado: predicción robusta
```

### Ventaja Principal
```
Reduce OVERFITTING automáticamente
└─ Árbol solo: Muy propenso a memorizar
└─ Random Forest: Muy robusto
```

### Feature Importance
```
¿Cuál es la característica más importante?

En UniversalBank:
1. Income:        0.3338 (33.38%) ← La más importante
2. Education:     0.2008 (20.08%)
3. CCAvg:         0.1721 (17.21%)
4. Family:        0.1114 (11.14%)
5. Age:           0.0363 (3.63%)   ← Menos importante

Interpretación: Income explica 33% de la predicción
```

---

## 4️⃣ VARIABLES DISCRETAS vs CONTINUAS

### Tabla Rápida
```
┌──────────────────┬──────────────┬──────────────────┐
│ Tipo de Modelo   │ Discretas    │ Continuas        │
├──────────────────┼──────────────┼──────────────────┤
│ MultinomialNB    │ ✅ Perfecto  │ ❌ Malo          │
│ GaussianNB       │ ❌ Malo      │ ✅ Perfecto      │
│ DecisionTree     │ ✅ Excelente │ ✅ Excelente     │
│ RandomForest     │ ✅ Excelente │ ✅ Excelente     │
└──────────────────┴──────────────┴──────────────────┘
```

### ¿Por qué Multinomial NB NECESITA discretas?
```
Si x es CONTINUA (ej: edad=34.567):
- Casi nunca se repite exactamente
- Conteos = 0 o 1 (no confiable)
- P(edad=34.567 | retraso) = imposible calcular

SOLUCIÓN: Binning/Categorización
edad = 34.567 → edad_30_40 = 1 (variable binaria)
Ahora: P(edad_30_40 | retraso) = fácil calcular ✅
```

### Cómo Convertir
```python
# Opción 1: Binning manual
df['edad_categoria'] = pd.cut(df['edad'], 
                               bins=[0, 30, 40, 50, 100])

# Opción 2: One-hot encoding (como en FlightDelays)
X = pd.get_dummies(df[['DAY_WEEK', 'CARRIER']])
# Resultado: DAY_WEEK_1, DAY_WEEK_2, ..., CARRIER_DL, etc.

# Opción 3: Label encoding (MALO para NB)
df['aerolínea_numero'] = df['CARRIER'].map({'AA': 0, 'DL': 1})
# ❌ Problema: Asume orden (1 > 0)
```

---

## 5️⃣ SELECCIÓN DE CARACTERÍSTICAS

### Cuatro Pasos
```
1. ¿CUÁLES SÍ usar?
   └─ Tienen relación lógica con objetivo
   └─ Información disponible en tiempo de predicción
   └─ Tienen poder predictivo

2. ¿CUÁLES NO usar?
   └─ Identificadores (ID, ZIP Code)
   └─ Información futura (no disponible)
   └─ Muy correlacionadas con otra

3. ¿CÓMO ELEGIRLAS?
   └─ Correlación con objetivo
   └─ Feature Importance de árboles
   └─ Mutual Information

4. ¿CUÁNTAS?
   └─ Menos = modelo más simple y rápido
   └─ Más = mejor precisión (hasta cierto punto)
```

### En Nuestros Notebooks

**FlightDelays - Multinomial NB:**
```python
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST', 'CARRIER']

¿POR QUÉ?
✅ Disponibles ANTES del vuelo
✅ Todas categóricas/discretas
✅ Tienen poder predictivo
✅ Fáciles de obtener

¿POR QUÉ NO otras?
❌ DEP_TIME: Solo disponible después
❌ Weather: Podría usarse pero no estaba
❌ TAIL_NUM: Muy específica del avión
```

**UniversalBank - Decision Trees:**
```
✅ Todas las columnas numéricas (Age, Income, etc.)
❌ EXCLUIDAS: ID, ZIP Code, Personal Loan

Ranked por importancia:
1. Income (0.334) → Capacidad de pago
2. Education (0.201) → Ingresos futuros
3. CCAvg (0.172) → Comportamiento
4. Family (0.111) → Necesidades
5. Experience (0.036) → Estabilidad
```

---

## 6️⃣ MATRIZ DE CONFUSIÓN RÁPIDA

### El Cuadro
```
                  Predicho
             Positive  Negative
Real Positive  TP      FN
Real Negative  FP      TN

Ejemplo: Predicción de retrasos
                  Predicho
             Delay  OnTime
Real Delay    45      10  ← Encontré 45 retrasos
Real OnTime   15     280  ← Predije que 280 llegarían a tiempo
```

### Métricas de 30 Segundos
```
Accuracy   = (TP+TN)/(Total) ← Porcentaje correcto (general)
Precision  = TP/(TP+FP)      ← De mis predicciones delay, ¿cuántas correctas?
Recall     = TP/(TP+FN)      ← De los retrasos reales, ¿cuántos detecté?
```

### En nuestros notebooks
```
FlightDelays (Multinomial NB):
└─ Accuracy: 80.48% ✅ (bastante bueno)

UniversalBank (Random Forest):
└─ Accuracy: 98.20% ✅ (excelente)
```

---

## 7️⃣ VALIDACIÓN CRUZADA (CRUCIAL)

### ¿Por qué?
```
SIN Cross-Validation:
├─ Split random train/test
├─ Resultado depende de la suerte
└─ No confiable ❌

CON 5-Fold CV:
├─ 5 divisiones diferentes
├─ Promedio de 5 resultados
└─ Robusto y confiable ✅
```

### Visualización
```
Dataset total = 1000 muestras

Fold 1: Train [200-1000] Test [0-200]     Score₁ = 0.85
Fold 2: Train [0-200, 400-1000] Test [200-400]   Score₂ = 0.84
Fold 3: Train [0-400, 600-1000] Test [400-600]   Score₃ = 0.86
Fold 4: Train [0-600, 800-1000] Test [600-800]   Score₄ = 0.85
Fold 5: Train [0-800] Test [800-1000]            Score₅ = 0.84

PROMEDIO = (0.85 + 0.84 + 0.86 + 0.85 + 0.84) / 5 = 0.848 ✅
```

---

## 8️⃣ GRIDSEARCHCV - Encontrar Mejores Parámetros

### El Concepto
```
Probar todas las combinaciones de parámetros
y elegir la mejor

Ejemplo del notebook:
param_grid = {
    'max_depth': [10, 20, 30, 40],           (4 valores)
    'min_samples_split': [20, 40, 60, ...],  (5 valores)
    'min_impurity_decrease': [0, 0.0005, ...] (5 valores)
}

Combinaciones = 4 × 5 × 5 = 100 ← probar 100 modelos diferentes
```

### Resultado
```
Mejor score encontrado: 0.9877 (98.77%)
Mejores parámetros: {
    'max_depth': 4,
    'min_impurity_decrease': 0.0011,
    'min_samples_split': 13
}
```

---

## 9️⃣ FÓRMULAS PARA MEMORIZAR

```
PROBABILIDAD POSTERIOR (Naive Bayes):
P(y|X) ∝ P(X|y) × P(y)

GINI (Árbol):
Gini = 1 - Σⱼ pⱼ²

INFORMACIÓN GAIN (Árbol):
IG = Gini_padre - Media(Gini_hijos)

LAPLACE SMOOTHING (Naive Bayes):
P = (conteos + α) / (total + α×V)

ACCURACY:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

PRECISION:
Precision = TP / (TP + FP)

RECALL:
Recall = TP / (TP + FN)

F1-SCORE:
F1 = 2 × (Precision × Recall) / (P + R)
```

---

## 🔟 TABLA COMPARATIVA FINAL

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Criterio            │ NB Multinomial│ Decision Tree│Random Forest │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Tipo de datos ideal │ Discretos    │ Mixtos       │ Mixtos       │
│ Velocidad train     │ ⚡⚡⚡ Rápido │ ⚡⚡ Medio   │ 🐢 Lento     │
│ Interpretabilidad   │ ⭐⭐⭐ Alta  │ ⭐⭐⭐⭐ Muy │ ⭐⭐ Media   │
│ Overfitting risk    │ ✅ Bajo      │ ❌ Alto      │ ✅ Muy bajo  │
│ Feature Importance  │ ❌ No        │ ✅ Sí        │ ✅ Excelente │
│ Comple interacciones│ ❌ No        │ ✅ Sí        │ ✅ Sí        │
│ Datos continuos     │ ❌ Pobre     │ ✅ Excelente │ ✅ Excelente │
└─────────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 🎓 PREGUNTAS MÁS PROBABLES EN EXAMEN

### P1: ¿Por qué usar variables DISCRETAS en Naive Bayes?
```
RESPUESTA:
Porque MultinomialNB calcula P(x|y) = (conteos)/(total)

Si x es continua (ej: 34.567), no se repite exactamente
→ conteos = 0 ó 1 → estimación no confiable

Solución: Convertir a discreta (binning o one-hot encoding)
```

### P2: ¿Cuál es la diferencia entre GINI e INFORMACIÓN GAIN?
```
RESPUESTA:
- Gini: Mide IMPUREZA de un nodo (0=puro, 0.5=impuro)
- Information Gain: REDUCCIÓN en impureza tras dividir
- Se elige la división con mayor IG (mayor reducción)
```

### P3: ¿Cómo EVITAR OVERFITTING en árboles?
```
RESPUESTA:
1. Limitar profundidad (max_depth)
2. Requerir mínimas muestras para dividir (min_samples_split)
3. Requerir mejora significativa (min_impurity_decrease)
4. Usar Random Forest (reduce automáticamente)
5. Validación cruzada (medir generalización)
```

### P4: ¿Cuándo es MEJOR Random Forest que un solo árbol?
```
RESPUESTA:
- Cuando queremos mejor GENERALIZACIÓN
- Cuando queremos REDUCIR VARIANZA
- Cuando el dataset es GRANDE
- Cuando NO necesitamos máxima INTERPRETABILIDAD
- VENTAJA: 98% de precisión típicamente
- DESVENTAJA: Menos interpretable
```

### P5: ¿Qué es LAPLACE SMOOTHING y por qué usarlo?
```
RESPUESTA:
- Suma α (alpha, típicamente 1 o 0.01) al numerador y denominador
- Evita probabilidades CERO
- Permite generalizar a nuevas observaciones
- En Naive Bayes: alpha=0.01 en nuestro notebook
```

---

## ⚠️ ERRORES COMUNES EN EXAMEN

```
❌ NUNCA:
- Confundir Gini con Información Gain
- Usar discretas en Gaussian NB
- Usar continuas directamente en Multinomial NB
- Olvidar dividir datos en train/test
- Comparar modelos sin Cross-Validation

✅ SIEMPRE:
- Verificar tipo de variable (discreta/continua)
- Elegir algoritmo según datos
- Usar train/test split (70/30 o 80/20)
- Validar con Cross-Validation
- Reportar múltiples métricas (Accuracy, Precision, Recall)
```

---

## 📝 CHECKLIST ANTES DEL EXAMEN

```
□ ¿Entiendo por qué Naive Bayes usa discretas?
□ ¿Sé calcular Gini manualmente?
□ ¿Entiendo cómo funciona Decision Tree?
□ ¿Sé qué es overfitting y cómo evitarlo?
□ ¿Conozco los parámetros clave (max_depth, min_samples)?
□ ¿Entiendo Feature Importance en Random Forest?
□ ¿Puedo leer una Matriz de Confusión?
□ ¿Sé cuándo usar cada algoritmo?
□ ¿Entiendo One-Hot Encoding?
□ ¿Conozco Cross-Validation?
```

---

## 🚀 ACCIONES FINALES ANTES DEL EXAMEN

```
5 MINUTOS ANTES:
1. Repasa las fórmulas clave (están arriba)
2. Repasa cuándo usar cada algoritmo
3. Recuerda: NB = discretas, Tree = cualquiera

DURANTE EL EXAMEN:
1. Lee bien las preguntas
2. Dibuja árboles/matrices si es necesario
3. Muestra desarrollo matemático
4. Explica por qué elegiste cada algoritmo

SI NO SABES UNA PREGUNTA:
1. No entres en pánico
2. Escribe lo que sí sepas
3. Haz conexiones con conceptos relacionados
```

---

**¡ÉXITO EN TU EXAMEN! 🎉**

*Documento: Resumen Rápido para Examen*  
*Minería de Datos - 8vo Ciclo*
