# 📚 FEEDBACK COMPLETO: MULTINOMIAL NAIVE BAYES PARA PREDICCIÓN DE RETRASOS DE VUELOS

## 🎯 RESUMEN EJECUTIVO

El código `code_8_naiveBayes.txt` implementa correctamente un clasificador **Multinomial Naive Bayes** para predecir si un vuelo llegará a tiempo ("ontime") o con retraso ("delayed"). Aquí te explicamos cada parte y qué deberías analizar.

---

## 📋 ESTRUCTURA DEL CÓDIGO (Paso a Paso)

### **SECCIÓN 1: IMPORTACIONES**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from dmba import classificationSummary, gainsChart
```

**¿Qué importamos?**
- `pandas`: Manipulación de datos
- `train_test_split`: Dividir datos en train/validation
- `MultinomialNB`: El clasificador Naive Bayes
- `dmba`: Librerías para análisis de clasificación

**Para el examen:** Debes entender qué hace cada librería.

---

### **SECCIÓN 2: CARGA Y TRANSFORMACIÓN DE DATOS**

#### 2.1 - Convertir a Categorías
```python
delays_df.DAY_WEEK = delays_df.DAY_WEEK.astype('category')
delays_df['Flight Status'] = delays_df['Flight Status'].astype('category')
```
**¿Por qué?** Naive Bayes funciona mejor con datos categóricos/discretos (a diferencia de SparsePCA que necesita continuos).

#### 2.2 - Crear Bins Horarios
```python
delays_df.CRS_DEP_TIME = [round(t / 100) for t in delays_df.CRS_DEP_TIME]
```
**Ejemplo:** `1455` → `round(1455/100)` = `15` (15:00 horas)

**¿Por qué?** Convertimos horas en formato 24h a categorías discretas (0-23 horas).

#### 2.3 - One-Hot Encoding
```python
X = pd.get_dummies(delays_df[predictors])
```
**¿Qué es?** Convierte variables categóricas en variables binarias.

**Ejemplo:**
```
CARRIER original: [DL, AA, OH, ...]
        ↓
CARRIER_AA  CARRIER_DL  CARRIER_OH
    0           1           0
    1           0           0
    0           0           1
```

**Importancia:** MultinomialNB necesita features numéricas (0/1), no textos.

#### 2.4 - División Train/Test
```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=1)
```
- **60% training**: Para ajustar el modelo
- **40% validation**: Para evaluar desempeño en datos nuevos

---

### **SECCIÓN 3: ENTRENAMIENTO DEL MODELO**

```python
delays_nb = MultinomialNB(alpha=0.01)
delays_nb.fit(X_train, y_train)
```

**Parámetro `alpha=0.01`:**
- Es **Laplace Smoothing**
- Evita probabilidades cero (problema: si un valor nunca aparece en training, P=0 siempre)
- Valores más altos = suavizado más agresivo

---

### **SECCIÓN 4: PREDICCIONES**

#### 4.1 - Predicción de Clases
```python
y_valid_pred = delays_nb.predict(X_valid)
```
Retorna: `['ontime', 'delayed', 'delayed', 'ontime', ...]`

#### 4.2 - Predicción de Probabilidades
```python
predProb_valid = delays_nb.predict_proba(X_valid)
```
Retorna una matriz con:
```
      ontime  delayed
0     0.75    0.25      ← 75% probabilidad ontime
1     0.30    0.70      ← 70% probabilidad delayed
2     0.92    0.08
...
```

---

### **SECCIÓN 5: ANÁLISIS DE PROBABILIDADES CONDICIONALES**

```python
for predictor in predictors:
    df = train_df[['Flight Status', predictor]]
    freqTable = df.pivot_table(index='Flight Status', columns=predictor, aggfunc=len)
    propTable = freqTable.apply(lambda x: x / sum(x), axis=1)
    print(propTable)
```

**¿Qué hace?**

Crea una tabla como esta (Ej: DAY_WEEK):
```
           1      2      3      4      5      6      7
ontime   0.65   0.64   0.62   0.63   0.65   0.60   0.58
delayed  0.35   0.36   0.38   0.37   0.35   0.40   0.42
```

**Interpretación:**
- Domingo (7): Solo 58% de vuelos llegan a tiempo → MÁS probable retraso
- Lunes (1): 65% de vuelos a tiempo → MENOS probable retraso

**Para el examen:** Estos son los **priors condicionales** que usa Naive Bayes.

---

### **SECCIÓN 6: PREDICCIÓN PARA UN CASO ESPECÍFICO**

```python
mask = ((X_valid.CARRIER_DL == 1) & (X_valid.DAY_WEEK_7 == 1) &
        (X_valid.CRS_DEP_TIME_10 == 1) & (X_valid.DEST_LGA == 1) &
        (X_valid.ORIGIN_DCA == 1))
df[mask]
```

**Busca vuelos con:**
- Aerolínea: Delta (DL)
- Día: Domingo (7)
- Hora: 10:00 AM
- Destino: LaGuardia (LGA)
- Origen: Reagan National (DCA)

**Resultado:** Una tabla con todos los vuelos que coinciden + sus predicciones.

---

## 🧠 ¿QUÉ ES MULTINOMIAL NAIVE BAYES?

### **El Teorema de Bayes:**
```
P(Clase | Datos) = P(Datos | Clase) × P(Clase) / P(Datos)
```

**En nuestro caso:**
```
P(ontime | CARRIER=DL, DAY_WEEK=7, ...) 
= P(CARRIER=DL | ontime) × P(DAY_WEEK=7 | ontime) × ... × P(ontime)
```

### **"Ingenuo" (Naive):**
Asume que **todas las variables son independientes**.
- Realidad: No son independientes (ej: algunas aerolíneas en ciertos días)
- Pero funciona bien en la práctica de todas formas

### **"Multinomial":**
Trabaja con **datos categóricos** (múltiples categorías), no continuos.

---

## 📊 ¿QUÉ DEBERÍA ANALIZAR?

### **1. Distribución de Clases**
```python
train_df['Flight Status'].value_counts() / len(train_df)
```
- ¿Es balanceado? (50% ontime, 50% delayed?)
- ¿O desbalanceado? (80% ontime, 20% delayed?)

### **2. Probabilidades Condicionales por Predictor**

**Para DAY_WEEK:**
¿Qué días tienen más retrasos?

**Para ORIGIN/DEST:**
¿Hay aeropuertos con más retrasos?

**Para CARRIER:**
¿Hay aerolíneas más puntuales?

**Para CRS_DEP_TIME:**
¿Hay horas del día con más retrasos?

### **3. Matriz de Confusión**
```
           Predicho ontime  |  Predicho delayed
Real ontime      TP         |       FN
Real delayed     FP         |       TN
```

**Métricas clave:**
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

### **4. Comparación Train vs Validation**
- ¿El modelo se ajusta bien (no overfitting)?
- ¿Accuracy en train ≈ Accuracy en validation?

---

## 🎓 DIFERENCIA: SparsePCA vs Multinomial Naive Bayes

| Aspecto | SparsePCA | Multinomial NB |
|---------|-----------|----------------|
| **Tipo** | Reducción dimensional | Clasificación |
| **Datos** | Continuos preferentemente | Categóricos (después encoding) |
| **Objetivo** | Encontrar componentes principales | Predecir clase |
| **Uso** | Exploración/visualización | Predicción |

---

## 💡 TIPS PARA TU EXAMEN

### ✅ Debes saber explicar:
1. Por qué necesitamos `train_test_split`
2. Qué es `get_dummies` y por qué es necesario
3. Qué es el parámetro `alpha` en MultinomialNB
4. Cómo interpretar probabilidades condicionales
5. Diferencia entre `predict()` y `predict_proba()`

### ❌ Errores comunes a evitar:
- Usar datos continuos sin categorizar
- No dividir train/validation
- Olvidar one-hot encoding
- Confundir accuracy con recall/precision

### 🔍 Preguntas probables:
- "¿Qué predictor es más importante?"
- "¿Por qué retraso el domingo?" (ver tabla DAY_WEEK)
- "¿Cuál es la precisión del modelo?"
- "¿Qué significa P(delayed|CARRIER_DL)?"

---

## 📝 ANÁLISIS FINAL

El código es un **ejemplo completo y bien estructurado** de:
1. ✅ Preparación de datos categóricos
2. ✅ Entrenamiento de clasificador Naive Bayes
3. ✅ Evaluación de modelo
4. ✅ Análisis interpretable de resultados

Para tu examen, enfócate en **entender cada paso** y poder **explicar qué hace** cada línea de código.

---

**¡Éxito en tu examen!** 🚀

