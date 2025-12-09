# 📊 RESUMEN VISUAL: Multinomial Naive Bayes

## 🔄 FLUJO DEL ALGORITMO

```
┌─────────────────────────────────────────────────────────┐
│          FLUJO MULTINOMIAL NAIVE BAYES                  │
└─────────────────────────────────────────────────────────┘

1. DATOS CRUDOS
   ┌─────────────────────────┐
   │ FlightDelays.csv        │
   │ 2203 vuelos × 13 cols   │
   └────────────┬────────────┘
                │
                ▼
2. PREPARACIÓN DE DATOS
   ┌─────────────────────────────┐
   │ ✓ Categorizar variables     │
   │ ✓ Crear bins horarios       │
   │ ✓ One-hot encoding          │
   │ ✓ Seleccionar predictores   │
   └────────────┬────────────────┘
                │
                ▼
3. DIVISIÓN TRAIN/TEST (60/40)
   ┌──────────────────┬──────────────────┐
   │   TRAINING       │   VALIDATION     │
   │  1321 muestras   │  882 muestras    │
   └────────┬─────────┴────────┬─────────┘
            │                  │
            ▼                  │
4. ENTRENAMIENTO                │
   ┌──────────────────────────┐ │
   │ MultinomialNB(alpha=0.01)│ │
   │ - Calcula P(predictor|clase)
   │ - Calcula P(clase)       │ │
   │ - Almacena probabilidades│ │
   └────────┬─────────────────┘ │
            │                   │
            ▼                   │
5. PREDICCIÓN EN VALIDACIÓN     │
   ├─→ predict() → ['ontime', 'delayed', ...] 
   └─→ predict_proba() → [[0.75, 0.25], ...]
                         ◄──────────────────┘
            │
            ▼
6. EVALUACIÓN
   ├─ Accuracy: 80.5%
   ├─ Matriz de confusión
   └─ Análisis de errores
```

---

## 🧮 FÓRMULA MATEMÁTICA

```
                    P(x₁|y) × P(x₂|y) × ... × P(xₙ|y) × P(y)
P(y | x₁, x₂, ..., xₙ) = ──────────────────────────────────────
                               P(x₁, x₂, ..., xₙ)

Donde:
  y = clase (ontime / delayed)
  x₁, x₂, ..., xₙ = features (DAY_WEEK, CARRIER, etc.)
```

**Interpretación para nuestro caso:**
```
P(ontime | CARRIER=DL, DAY_WEEK=7, CRS_DEP_TIME=10, DEST=LGA, ORIGIN=DCA)
= P(DL|ontime) × P(7|ontime) × P(10|ontime) × P(LGA|ontime) × P(DCA|ontime) × P(ontime)
```

---

## 📊 TRANSFORMACIÓN DE DATOS

### One-Hot Encoding Ejemplo

```
ORIGINAL:                 ONE-HOT ENCODED:
┌──────────┐             ┌─────────────────────────┐
│ CARRIER  │      →      │ CARRIER_AA  CARRIER_DL  │
├──────────┤             ├─────────────────────────┤
│   DL     │             │    0            1       │
│   AA     │             │    1            0       │
│   DL     │             │    0            1       │
└──────────┘             └─────────────────────────┘
```

### Bins Horarios

```
ORIGINAL TIME: 1455 (14:55)
       ↓
   1455 / 100 = 14.55
       ↓
   round(14.55) = 15 horas (3 PM)
       ↓
CATEGORIZADO: 15
```

---

## 🎯 TABLA DE PROBABILIDADES CONDICIONALES

**Ejemplo: ¿Qué día tiene más retrasos?**

```
                DAY_WEEK (1=Lunes, 7=Domingo)
             ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
             │  1   │  2   │  3   │  4   │  5   │  6   │  7   │
┌────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ P(ontime)  │ 0.65 │ 0.64 │ 0.62 │ 0.63 │ 0.65 │ 0.60 │ 0.58 │ ← Domingo tiene MENOS
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ P(delayed) │ 0.35 │ 0.36 │ 0.38 │ 0.37 │ 0.35 │ 0.40 │ 0.42 │ ← Domingo tiene MÁS
└────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

CONCLUSIÓN: Vuelos el domingo (7) tienen MAYOR probabilidad de retraso (42%)
```

---

## 🔍 PREDICCIÓN PASO A PASO

```
NUEVO VUELO:
┌──────────────────────────────┐
│ CARRIER: Delta (DL)          │
│ DAY_WEEK: Domingo (7)        │
│ CRS_DEP_TIME: 15:00 (15 hrs) │
│ DEST: LaGuardia (LGA)        │
│ ORIGIN: Reagan (DCA)         │
└──────────┬───────────────────┘
           │
           ▼
MODEL PREDICTIONS:
┌──────────────────────────────┐
│ P(ontime) = 0.35             │ (35% posibilidad)
│ P(delayed) = 0.65            │ (65% posibilidad)
└──────────┬───────────────────┘
           │
           ▼
PREDICCIÓN FINAL:
┌──────────────────┐
│ CLASE: DELAYED   │ ← Probabilidad más alta
└──────────────────┘
```

---

## ✅ MATRIZ DE CONFUSIÓN

```
                    PREDICCIÓN
                 ontime  delayed
             ┌──────────┬─────────┐
REAL ontime  │   TP=400 │ FN=100  │  (500 reales ontime)
             ├──────────┼─────────┤
     delayed │   FP=50  │ TN=332  │  (382 reales delayed)
             └──────────┴─────────┘

Cálculos:
├─ Accuracy = (TP + TN) / Total = (400 + 332) / 882 = 83%
├─ Precision = TP / (TP + FP) = 400 / 450 = 88.9%
└─ Recall = TP / (TP + FN) = 400 / 500 = 80%
```

---

## 🆚 COMPARACIÓN: SparsePCA vs Multinomial NB

```
┌─────────────────────┬────────────────┬─────────────────────┐
│ CARACTERÍSTICA      │  SparsePCA     │ Multinomial Naive   │
│                     │                │      Bayes          │
├─────────────────────┼────────────────┼─────────────────────┤
│ Tipo                │ Reducción dim. │ Clasificación       │
│ Input               │ Continuos      │ Categóricos (ideales)
│ Output              │ Componentes    │ Predicciones        │
│ Objetivo            │ Exploración    │ Predicción          │
│ Datos discretos OK  │ ❌ NO         │ ✅ SÍ              │
│ Complejidad         │ Media          │ Baja                │
│ Interpretable       │ Medio          │ Alto                │
└─────────────────────┴────────────────┴─────────────────────┘
```

---

## 🚨 PROBLEMAS COMUNES Y SOLUCIONES

```
┌────────────────────────────────────────────────────────┐
│ PROBLEMA: Probabilidades cero                          │
├────────────────────────────────────────────────────────┤
│ CAUSA: Un valor nunca aparece en training             │
│ EJEMPLO: Si aerolínea 'XYZ' no aparece en train,      │
│          P(XYZ) = 0 siempre                           │
│ SOLUCIÓN: Usar alpha > 0 (Laplace Smoothing)         │
│ EN CÓDIGO: MultinomialNB(alpha=0.01)                 │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ PROBLEMA: Overfitting                                 │
├────────────────────────────────────────────────────────┤
│ CAUSA: Entrenar y probar con MISMOS datos             │
│ RESULTADO: Accuracy alto en train, bajo en test       │
│ SOLUCIÓN: Dividir en train/validation (60/40)        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ PROBLEMA: Variables categóricas vs numéricas          │
├────────────────────────────────────────────────────────┤
│ CAUSA: MultinomialNB espera números, recibe textos    │
│ SOLUCIÓN: One-hot encoding con pd.get_dummies()      │
└────────────────────────────────────────────────────────┘
```

---

## 💡 CONCEPTOS CLAVE EXPLICADOS

### **¿Qué es "Naive"?**
```
Naive = INGENUO

Se llama "ingenuo" porque ASUME que todos los
predictores son INDEPENDIENTES entre sí:

Realidad:          Asunción Naive:
CARRIER_DL ───┐   CARRIER_DL ┬ independiente
              ├─ Correlacionadas
DAY_WEEK_7 ───┘               │
              └─ DAY_WEEK_7 independiente

Aunque la asunción es FALSA, el algoritmo funciona
sorprendentemente bien en la práctica.
```

### **¿Qué es Laplace Smoothing (alpha)?**
```
SIN Smoothing (alpha=0):
  P(nunca_visto) = 0 ← PROBLEMA: No puedes predecir

CON Smoothing (alpha=0.01):
  P(nunca_visto) = 0.01 ← OK: Pequeña probabilidad
  
Más alto alpha → Más suavizado
  alpha=1.0 → Muy suavizado
  alpha=0.01 → Poco suavizado
```

### **¿Qué es One-Hot Encoding?**
```
ANTES:          DESPUÉS:
CARRIER         CARRIER_AA  CARRIER_DL  CARRIER_OH
DL      →            0           1           0
AA              →    1           0           0
OH              →    0           0           1

VENTAJA: MultinomialNB necesita números (0/1)
DESVENTAJA: Aumenta número de columnas
```

---

## 📋 CHECKLIST ANTES DEL EXAMEN

- [ ] Entiendo la fórmula del Teorema de Bayes
- [ ] Puedo explicar por qué "ingenuo"
- [ ] Sé qué es Laplace Smoothing
- [ ] Entiendo One-Hot Encoding
- [ ] Puedo leer una tabla de probabilidades condicionales
- [ ] Sé interpretar una matriz de confusión
- [ ] Entiendo por qué train/test split es importante
- [ ] Puedo explicar predict() vs predict_proba()
- [ ] Sé diferenciar SparsePCA de Multinomial NB
- [ ] Puedo ejecutar el código sin errores

---

## 🎯 PREGUNTAS DE EXAMEN PROBABLE

```
P1: ¿Por qué se llama "Naive"?
R: Porque asume independencia entre predictores

P2: ¿Qué hace pd.get_dummies()?
R: Convierte categorías en variables binarias (0/1)

P3: ¿Cuál es la diferencia entre predict() y predict_proba()?
R: predict() → clase ganadora
   predict_proba() → todas las probabilidades

P4: ¿Por qué importa train_test_split?
R: Para evitar overfitting y evaluar en datos nuevos

P5: ¿Qué es alpha en MultinomialNB(alpha=0.01)?
R: Laplace Smoothing para evitar P=0

P6: ¿Qué diferencia hay con SparsePCA?
R: SparsePCA explora (dimensionalidad), NB predice (clasificación)

P7: ¿Qué significa P(ontime | CARRIER=DL)?
R: Probabilidad de llegar a tiempo DADO que es Delta Airlines
```

---

**Ahora tienes una comprensión completa de Multinomial Naive Bayes** ✅

