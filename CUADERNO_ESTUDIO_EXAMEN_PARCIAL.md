# 📚 CUADERNO DE ESTUDIO - EXAMEN PARCIAL CC442
## Minería de Datos (8vo Ciclo)

**Fecha:** 2025  
**Profesor:** Basado en teoría del curso  
**Temas:** Naive Bayes, Regresión Lineal, Linear Discriminant Analysis

---

## 📋 ÍNDICE DEL CUADERNO

1. [Teoría de Algoritmos de Clasificación](#teoría)
2. [Problema 1: Automobile Accidents](#problema-1)
3. [Problema 2: Toyota Corolla Prices](#problema-2)
4. [Problema 3: Spam Detection](#problema-3)
5. [Resumen de Evaluación](#resumen)
6. [Preguntas de Examen Probable](#preguntas)

---

## <a name="teoría"></a>
# PARTE 1: TEORÍA FUNDAMENTAL

## 1.1 Naive Bayes - Fundamentos

### ¿Qué es Naive Bayes?

**Definición:** Algoritmo de clasificación probabilístico basado en el Teorema de Bayes que asume independencia condicional entre predictores.

**Fórmula clave:**
```
P(Clase | Características) ∝ P(Características | Clase) × P(Clase)
```

### ¿Por qué se llama "Naive" (Ingenuo)?

Porque **asume independencia entre predictores** aunque en realidad pueden estar correlacionados.

**Ejemplo:**
```
En un email:
- P(palabra "dinero" | spam)
- P(palabra "gratis" | spam)
- P(palabra "urgente" | spam)

Naive Bayes ASUME que estas palabras son independientes
P(dinero, gratis, urgente | spam) = P(dinero|spam) × P(gratis|spam) × P(urgente|spam)

En realidad, los spams que mencionan "dinero" también tienden a mencionar "gratis"
(correlacionadas), pero el algoritmo ignora esto.

¡Y funciona bien de todas formas! (sorprendentemente)
```

### Variantes de Naive Bayes

```
┌─────────────────────────────────────────────────────────────┐
│ TIPO                │ DATOS IDEALES      │ FÓRMULA         │
├─────────────────────────────────────────────────────────────┤
│ Multinomial NB      │ Datos discretos    │ Conteos         │
│ (FlightDelays)      │ Palabras/conteos   │ (para spam)     │
│                     │                    │                 │
│ Gaussian NB         │ Datos continuos    │ Distribución    │
│                     │ Distribuidos       │ gaussiana       │
│                     │ normalmente        │                 │
│                     │                    │                 │
│ Bernoulli NB        │ Variables binarias │ 0/1 solamente   │
│                     │ (presencia/ausencia)                 │
└─────────────────────────────────────────────────────────────┘
```

### ¿Por qué Multinomial NB necesita datos DISCRETOS?

**Problema con datos continuos:**

Si una variable es continua (e.g., edad = 34.567 años):
- En el dataset de entrenamiento, probablemente NO se repetirá exactamente 34.567
- El algoritmo contaría: 1 o 0 ocurrencias (muy poco confiable)
- P(edad=34.567 | retraso) sería imposible de calcular correctamente

**Solución - Convertir a discreta:**

```python
# OPCIÓN 1: Binning manual
df['edad_categoria'] = pd.cut(df['edad'], bins=[0, 18, 30, 45, 65, 100])
# Resultado: edad_0_18, edad_18_30, edad_30_45, edad_45_65, edad_65_100

# OPCIÓN 2: One-Hot Encoding (como en el examen)
X = pd.get_dummies(df[['WEATHER_R', 'TRAF_CON_R']])
# Resultado: WEATHER_R_0, WEATHER_R_1, WEATHER_R_2, TRAF_CON_R_0, etc.

# OPCIÓN 3: Usar GaussianNB (para datos continuos)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_continuo, y)
```

### Laplace Smoothing (Parámetro alpha)

**Problema sin smoothing:**

```
Si "CARRIER=XYZ" nunca aparece en los vuelos retrasados del training:
P(CARRIER=XYZ | delayed) = 0 / total = 0

Cuando llegue un vuelo real con XYZ:
P(delayed | XYZ) = 0 (¡nunca predecirá delayed!)
```

**Solución - Laplace Smoothing:**

```
P(característica | clase) = (conteos + α) / (total + α × V)

donde:
- α = parámetro de suavizado (típicamente 1.0 o 0.01)
- V = número de valores únicos de la característica

Ejemplo:
P(CARRIER=XYZ | delayed) = (0 + 0.01) / (1000 + 0.01×100) = 0.01 / 1001

¡Ahora NO es cero! (pequeña probabilidad que permite predicción)
```

---

## 1.2 Regresión Lineal Múltiple

### Concepto Fundamental

**Objetivo:** Predecir una variable continua (precio) usando múltiples predictores.

**Modelo:**
```
y = β₀ + β₁×x₁ + β₂×x₂ + ... + βₙ×xₙ + ε

donde:
- y = variable objetivo (Price)
- β₀ = intercepto
- β₁, β₂, ..., βₙ = coeficientes
- x₁, x₂, ..., xₙ = predictores
- ε = error residual
```

### Interpretación de Coeficientes

```
Si el coeficiente de "Edad" es -500:
"Cada año adicional de edad → DISMINUYE el precio en $500"

Si el coeficiente de "HP" es 100:
"Cada caballode fuerza adicional → AUMENTA el precio en $100"
```

### Evaluación del Modelo

**R² (Coeficiente de Determinación):**
```
R² = 1 - (SS_residual / SS_total)

Rango: 0 a 1 (o 0% a 100%)

Interpretación:
- R² = 0.85 → El modelo explica 85% de la varianza
- R² = 0.50 → Explicamos solo 50% (malo)
- R² = 0.95 → Explicamos 95% (excelente)
```

**RMSE (Root Mean Squared Error):**
```
RMSE = √(Σ(y_real - y_predicción)² / n)

Interpretación:
- Unidades: Mismas que la variable objetivo (ej: dólares)
- RMSE = $2000 → En promedio, nuestras predicciones están $2000 de error
```

---

## 1.3 Linear Discriminant Analysis (LDA)

### Concepto

**Objetivo:** Clasificación que busca encontrar una combinación lineal de características que mejor separe las clases.

**Idea visual:**

```
Sin LDA:                    Con LDA:
Clase A: ●●●             Proyectado en LD1:
Clase B: ○○○             ●●● | ○○○
         (datos           (mejor separación)
          mezclados)
```

### Ventajas de LDA

```
✓ Funciona bien con datos multidimensionales
✓ Computacionalmente eficiente
✓ Proporciona matriz de proyección
✓ Tiene en cuenta estructura de covarianza
✓ No requiere muchos parámetros de tuning
```

### Matriz de Confusión en Clasificación Binaria

```
                    Predicción
                Spam    No-Spam
    Real Spam      TP        FN
    Real No-Spam   FP        TN

Donde:
- TP (True Positive): Predijo spam, era spam ✓
- FP (False Positive): Predijo spam, era no-spam ✗ (falsa alarma)
- FN (False Negative): Predijo no-spam, era spam ✗ (se coló el spam)
- TN (True Negative): Predijo no-spam, era no-spam ✓
```

### Métricas de Evaluación

```
Accuracy = (TP + TN) / Total
→ Porcentaje total correcto

Precision = TP / (TP + FP)
→ De los que predicho como SPAM, ¿cuántos realmente lo eran?
→ Evita falsas alarmas

Recall (Sensitivity) = TP / (TP + FN)
→ De los SPAM reales, ¿cuántos detecté?
→ Capacidad de detectar positivos

Specificity = TN / (TN + FP)
→ De los NO-SPAM, ¿cuántos identifico correctamente?
```

---

# <a name="problema-1"></a>
# PROBLEMA 1: AUTOMOBILE ACCIDENTS

## Objetivo

Predecir si un accidente automovilístico involucrará lesión (INJURY = yes/no) usando Naive Bayes.

## Datos

```
Dataset: AccidentsFull.csv
- 42,183 registros
- Accidentes en USA, año 2001
- Información: clima, tráfico, carretera, etc.

Variable objetivo:
- MAX_SEV_IR: 0=NO INJURY, 1=INJURY, 2=FATALITY
- Transformar a: INJURY = "yes" si MAX_SEV_IR ∈ {1,2}, "no" si MAX_SEV_IR = 0
```

## Solución Paso a Paso

### PASO 1.1: Primeros 12 registros

**Que hacer:**
1. Cargar AccidentsFull.csv
2. Crear variable INJURY
3. Seleccionar primeros 12 registros
4. Usar solo WEATHER_R y TRAF_CON_R como predictores
5. Entrenar Multinomial Naive Bayes
6. Mostrar probabilidades y predicciones
7. Usar threshold = 0.5

**Código:**

```python
# Cargar y preparar
accidents_df = pd.read_csv('AccidentsFull.csv')
accidents_df['INJURY'] = accidents_df['MAX_SEV_IR'].apply(
    lambda x: 'yes' if x in [1, 2] else 'no'
)

# Seleccionar primeros 12
sample_12 = accidents_df[['INJURY', 'WEATHER_R', 'TRAF_CON_R']].head(12)

# One-hot encoding
X_12 = pd.get_dummies(sample_12[['WEATHER_R', 'TRAF_CON_R']])
y_12 = sample_12['INJURY']

# Entrenar
nb = MultinomialNB(alpha=0.01)
nb.fit(X_12, y_12)

# Predicciones
proba = nb.predict_proba(X_12)
pred = nb.predict(X_12)

# Mostrar resultados
for i in range(len(X_12)):
    print(f"{i}: P(no)={proba[i,0]:.4f}, P(yes)={proba[i,1]:.4f}, Pred={pred[i]}")
```

**Interpretación esperada:**
- Para cada registro mostrará probabilidades
- Si P(yes) >= 0.5 → predecir "yes"
- Si P(yes) < 0.5 → predecir "no"
- Comparar con valores reales

### PASO 1.2: Análisis completo

**1.2.1 Predictores disponibles:**

```
✓ Disponibles ANTES de reportar el accidente:
- WEATHER_R: Condiciones climáticas
- TRAF_CON_R: Condiciones de tráfico
- LGTCON_I_R: Condiciones de luz
- WKDY_I_R: Día de la semana
- HOUR_I_R: Hora del accidente
- INT_HWY: Tipo de carretera
- SPD_LIM: Límite de velocidad
```

**1.2.2 Entrenar el modelo:**

```python
# División 60/40
X_full = pd.get_dummies(accidents_df[['WEATHER_R', 'TRAF_CON_R']])
y_full = accidents_df['INJURY']

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.4, random_state=1, stratify=y_full
)

# Entrenar
nb_model = MultinomialNB(alpha=0.01)
nb_model.fit(X_train, y_train)

# Evaluar
y_pred = nb_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred, labels=['no', 'yes'])

# Mostrar matriz
print("Matriz de Confusión:")
print(cm)

# Calcular accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Error total
error = 1 - accuracy
print(f"Error Total: {error:.4f}")
```

**Interpretación:**

```
Matriz de Confusión:
           Predicho
       no    yes
Real no  TN    FP
Real yes FN    TP

Accuracy = (TN + TP) / (Total)
→ Porcentaje de predicciones correctas

Error = 1 - Accuracy
→ Porcentaje de predicciones incorrectas
```

---

# <a name="problema-2"></a>
# PROBLEMA 2: TOYOTA COROLLA PRICES

## Objetivo

Predecir el precio de un Toyota Corolla usado usando Regresión Lineal Múltiple.

## Datos

```
Dataset: ToyotaCorolla.csv
- 1,436 registros
- 38 atributos
- Año: 2004 (Países Bajos)

Predictores obligatorios:
Age_08_04, KM, Fuel_Type, HP, Automatic, Doors,
Quarterly_Tax, Mfr_Guarantee, Guarantee_Period,
Airco, Automatic_airco, CD_Player, Powered_Windows,
Sport_Model, Tow_Bar

Variable objetivo: Price
```

## Solución

### PASO 2.1: División y Modelo

```python
# Cargar
toyota_df = pd.read_csv('ToyotaCorolla.csv')

# Predictores
predictores = ['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Automatic', 'Doors',
               'Quarterly_Tax', 'Mfr_Guarantee', 'Guarantee_Period',
               'Airco', 'Automatic_airco', 'CD_Player', 'Powered_Windows',
               'Sport_Model', 'Tow_Bar']

# Limpiar NaN
toyota_clean = toyota_df[['Price'] + predictores].dropna()

# One-hot encoding para Fuel_Type
toyota_encoded = pd.get_dummies(toyota_clean, columns=['Fuel_Type'], drop_first=True)
X_toyota = toyota_encoded.drop('Price', axis=1)
y_toyota = toyota_encoded['Price']

# División 70/30
X_train, X_val, y_train, y_val = train_test_split(
    X_toyota, y_toyota, test_size=0.3, random_state=1
)

# Entrenar
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluar
r2_train = lr.score(X_train, y_train)
r2_val = lr.score(X_val, y_val)
rmse_val = np.sqrt(mean_squared_error(y_val, lr.predict(X_val)))

print(f"R² Train: {r2_train:.4f}")
print(f"R² Val: {r2_val:.4f}")
print(f"RMSE Val: ${rmse_val:,.2f}")
```

### PASO 2.2: Análisis de Importancia

```python
# Extraer coeficientes
coef_df = pd.DataFrame({
    'Predictor': X_toyota.columns,
    'Coeficiente': lr.coef_,
    'Abs_Coef': np.abs(lr.coef_)
}).sort_values('Abs_Coef', ascending=False)

# Top predictores
print("TOP 3-4 PREDICTORES MÁS IMPORTANTES:")
print(coef_df.head(4))

# Interpretación
for idx, row in coef_df.head(4).iterrows():
    print(f"\n{row['Predictor']}:")
    print(f"  Coeficiente: {row['Coeficiente']:,.2f}")
    if row['Coeficiente'] > 0:
        print(f"  → AUMENTA precio: +${abs(row['Coeficiente']):,.2f} por unidad")
    else:
        print(f"  → DISMINUYE precio: -${abs(row['Coeficiente']):,.2f} por unidad")
```

### Interpretación esperada:

```
Los predictores más importantes son probablemente:
1. Age (edad): Coef < 0 → Más viejo = menos precio
2. KM (kilómetros): Coef < 0 → Más uso = menos precio
3. HP (potencia): Coef > 0 → Más potencia = más precio
4. Automatic (automático): Coef ? → Depende del mercado
```

---

# <a name="problema-3"></a>
# PROBLEMA 3: SPAM DETECTION

## Objetivo

Detectar emails de spam vs no-spam usando Linear Discriminant Analysis con los 11 predictores más discriminativos.

## Datos

```
Dataset: spambase.csv
- 4,601 emails
- 57 predictores (frecuencias de palabras/símbolos)
- 1,813 spam (39.4%), 2,788 no-spam (60.6%)

Última columna: Spam (0=no-spam, 1=spam)
```

## Solución

### PASO 3.1: Seleccionar 11 predictores

**Idea:** Encontrar palabras/símbolos que MÁS diferencian spam de no-spam

```python
# Cargar
spam_df = pd.read_csv('spambase.csv')
X_spam = spam_df.drop('Spam', axis=1)
y_spam = spam_df['Spam']

# Calcular diferencias de medias
spam_class = X_spam[y_spam == 1].mean()
nonspam_class = X_spam[y_spam == 0].mean()

diff = np.abs(spam_class - nonspam_class).sort_values(ascending=False)

# Top 11
top_11 = diff.head(11)

print("TOP 11 PREDICTORES QUE DIFERENCIAN SPAM DE NO-SPAM:")
for i, (feat, val) in enumerate(top_11.items(), 1):
    mean_spam = spam_class[feat]
    mean_nonspam = nonspam_class[feat]
    print(f"{i:2d}. {feat:20s} | Spam: {mean_spam:.4f} | No-Spam: {mean_nonspam:.4f} | Diff: {val:.4f}")
```

**Interpretación:**

```
Palabras que aparecen MÁS en SPAM:
- "money", "free", "business", "http", "email"

Palabras que aparecen MÁS en NO-SPAM:
- "george", "data", "lab", "address"

Símbolos que aparecen MÁS en SPAM:
- "$", "!", "(", "[", ";"
```

### PASO 3.2: Entrenar LDA

```python
# Seleccionar 11 predictores
top_11_preds = diff.head(11).index.tolist()
X_spam_selected = X_spam[top_11_preds]

# División
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_spam_selected, y_spam, test_size=0.3, random_state=1, stratify=y_spam
)

# Entrenar LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_s, y_train_s)

# Predicciones
y_pred_s = lda.predict(X_val_s)

# Matriz de confusión
cm = confusion_matrix(y_val_s, y_pred_s)
print("Matriz de Confusión:")
print(cm)

# Métricas
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
```

### PASO 3.3: Evaluación del Modelo

**¿Es útil el modelo?**

```
Criterios de utilidad:
✓ Accuracy > 70%:    Modelo funciona básicamente
✓ Accuracy > 85%:    Modelo bueno
✓ Accuracy > 95%:    Modelo excelente

✓ Precision > 90%:   Pocos falsos positivos (emails legítimos marcados)
✓ Recall > 85%:      Capturamos la mayoría del spam

RESPUESTA ESPERADA:
"Este modelo IS/NO útil para detectar spam porque:
- Accuracy = X% (suficiente/insuficiente)
- Precision = X% (pocas/muchas falsas alarmas)
- Recall = X% (detectamos mucho/poco spam real)"
```

---

# <a name="resumen"></a>
# RESUMEN DE EVALUACIÓN

## Rubrica del Examen

```
PROBLEMA 1: AUTOMOBILE ACCIDENTS (30 puntos)
├─ 1.1 Primeros 12 registros         (10 pts)
│  ├─ Cargar datos                   (2 pts)
│  ├─ Crear variable INJURY          (2 pts)
│  ├─ Aplicar Naive Bayes            (3 pts)
│  └─ Mostrar probabilidades         (3 pts)
│
└─ 1.2 Dataset completo              (20 pts)
   ├─ División 60/40                 (5 pts)
   ├─ Entrenar NB                    (5 pts)
   ├─ Matriz de confusión            (5 pts)
   └─ Calcular error total           (5 pts)

PROBLEMA 2: TOYOTA COROLLA (35 puntos)
├─ 2.1 Modelo de regresión           (20 pts)
│  ├─ División 70/30                 (5 pts)
│  ├─ Entrenar modelo                (5 pts)
│  ├─ Histogramas de errores         (5 pts)
│  └─ Interpretación                 (5 pts)
│
└─ 2.2 Predictores más importantes   (15 pts)
   ├─ Identificar top 3-4            (8 pts)
   └─ Justificación                  (7 pts)

PROBLEMA 3: SPAM DETECTION (35 puntos)
├─ 3.1 Seleccionar 11 predictores    (12 pts)
│  ├─ Calcular diferencias           (6 pts)
│  └─ Identificar palabras/símbolos  (6 pts)
│
├─ 3.2 Entrenar LDA                  (13 pts)
│  ├─ División datos                 (3 pts)
│  ├─ Modelo LDA                     (5 pts)
│  └─ Predicciones                   (5 pts)
│
└─ 3.3 Evaluación del modelo         (10 pts)
   ├─ Matriz de confusión            (5 pts)
   └─ Interpretación de utilidad     (5 pts)

TOTAL: 100 puntos
```

---

# <a name="preguntas"></a>
# PREGUNTAS DE EXAMEN PROBABLE

## Pregunta Teórica 1

**"¿Por qué es importante crear la variable INJURY (sí/no) en lugar de usar directamente MAX_SEV_IR (0, 1, 2)?"**

**Respuesta esperada:**
```
El problema requiere clasificación BINARIA (lesión sí/no), no multiclase.

MAX_SEV_IR = 0: Sin lesión
MAX_SEV_IR = 1: Lesión moderada
MAX_SEV_IR = 2: Fatality

Para predicción rápida en reportes iniciales, interesa saber:
¿Hay lesión o no? (sí/no)

No importa diferenciar entre lesión moderada y fatality en tiempo real.

INJURY = "yes" si MAX_SEV_IR ∈ {1, 2}
INJURY = "no" si MAX_SEV_IR = 0
```

## Pregunta Teórica 2

**"¿Por qué Multinomial Naive Bayes requiere datos discretos?"**

**Respuesta esperada:**
```
Multinomial Naive Bayes estima:

P(característica | clase) = (conteos de característica en clase) / (total de conteos)

Si la característica es CONTINUA (ej: edad = 34.567 años):
- Prácticamente nunca se repetirá exactamente 34.567 en validation
- Conteos en training = quizás 1 o 0
- Estimación = muy poco confiable
- P(edad=34.567 | retraso) = casi imposible de calcular

SOLUCIÓN: Convertir a discreta (binning) o usar GaussianNB
```

## Pregunta Práctica 1

**"¿Cuál fue la matrix de confusión para el problema de accidentes? Interprete cada valor."**

**Respuesta esperada:**
```
Matriz típica esperada:
         Predicho
       no    yes
Real no  TN    FP
Real yes FN    TP

Interpretación:
- TN (True Negative): Accidentes sin lesión predichos correctamente
- FP (False Positive): Sin lesión pero predijo que hay (falsa alarma)
- FN (False Negative): Con lesión pero predijo que no (riesgo!)
- TP (True Positive): Con lesión predichos correctamente

Error Total = (FP + FN) / Total

En contexto de seguridad: FN es más peligroso que FP
(mejor alarma falsa que no detectar lesión real)
```

## Pregunta Práctica 2

**"¿Cuáles fueron los 3 principales predictores del precio en Toyota Corolla? ¿Por qué?"**

**Respuesta esperada:**
```
Esperados (basado en lógica):

1. Age_08_04 (edad): Coef ≈ -500 a -1000
   → Autos más viejos valen MENOS
   → Depreciación obvia

2. KM (kilómetros): Coef ≈ -1 a -5
   → Más uso = menos precio
   → Desgaste acumulado

3. HP (potencia): Coef ≈ 50 a 200
   → Más potencia = más precio
   → Mayor capacidad = valor

Estos son intuitivos porque:
- Edad: Factor fundamental de depreciación
- Km: Indicador de desgaste
- HP: Indicador de calidad/prestaciones
```

## Pregunta Práctica 3

**"¿Fue útil el modelo LDA para detectar spam? Justifique."**

**Respuesta esperada:**
```
Criterios de utilidad:

SI es útil si:
✓ Accuracy > 80%
✓ Precision > 85% (pocos emails legítimos marcados como spam)
✓ Recall > 75% (detectamos mayoría de spam)

NO es útil si:
✗ Accuracy < 70%
✗ Precision < 70% (falsas alarmas altas)
✗ Recall < 60% (mucho spam se cuela)

Respuesta esperada:
"El modelo [SÍ/NO] es útil porque:
- Accuracy = X%
- Precision = X% (significado: de cada 100 emails que predije como spam, X eran reales)
- Recall = X% (significado: de cada 100 spam reales, detecté X)
- En conclusión: [beneficio/inconveniente]"
```

---

## Checklist Antes del Examen

```
□ Entiendo por qué Naive Bayes necesita datos discretos
□ Puedo crear variable INJURY correctamente
□ Sé cómo dividir datos en train/validation
□ Puedo interpretar una matriz de confusión
□ Conozco la diferencia entre Accuracy y Error
□ Entiendo R² e RMSE en regresión
□ Puedo identificar predictores importantes
□ Sé cómo calcular diferencias de medias
□ Entiendo cuándo un modelo de clasificación es útil
□ He practicado cada problema al menos una vez
```

---

**Documento de Estudio Completo - Examen Parcial CC442**  
**Minería de Datos (8vo Ciclo) - 2025**
