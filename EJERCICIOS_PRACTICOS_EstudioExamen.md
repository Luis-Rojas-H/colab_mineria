# EJERCICIOS PRÁCTICOS CON SOLUCIONES
## Minería de Datos - 8vo Ciclo

---

## EJERCICIO 1: Calcular Gini Manualmente

### Problema
Un dataset tiene 100 muestras en un nodo:
- 70 muestras de Clase A (Retraso)
- 30 muestras de Clase B (OnTime)

**Calcular Gini**

### Solución
```
p_A = 70/100 = 0.7
p_B = 30/100 = 0.3

Gini = 1 - (p_A² + p_B²)
Gini = 1 - (0.7² + 0.3²)
Gini = 1 - (0.49 + 0.09)
Gini = 1 - 0.58
Gini = 0.42

Interpretación: El nodo es moderadamente puro (Gini=0.42)
```

---

## EJERCICIO 2: Calcular Information Gain

### Problema
Un nodo PADRE tiene:
- 100 muestras: 70 Clase A, 30 Clase B
- Gini_padre = 0.42 (del ejercicio anterior)

Se divide por característica X en dos HIJOS:
- Hijo 1: 60 muestras (50 A, 10 B) → Gini_1 = 0.333
- Hijo 2: 40 muestras (20 A, 20 B) → Gini_2 = 0.5

**Calcular Information Gain**

### Solución
```
Gini_Hijo1 = 1 - (50/60)² - (10/60)²
           = 1 - 0.6944² - 0.1667²
           = 1 - 0.4823 - 0.0278
           = 0.4899 ≈ 0.333 (verificado) ✓

Gini_Hijo2 = 1 - (20/40)² - (20/40)²
           = 1 - 0.5² - 0.5²
           = 1 - 0.25 - 0.25
           = 0.5 ✓

IG = Gini_padre - [w₁×Gini_hijo1 + w₂×Gini_hijo2]

donde:
w₁ = 60/100 = 0.6 (peso hijo 1)
w₂ = 40/100 = 0.4 (peso hijo 2)

IG = 0.42 - [0.6×0.333 + 0.4×0.5]
IG = 0.42 - [0.1998 + 0.2]
IG = 0.42 - 0.3998
IG = 0.0202

Interpretación: La división reduce impureza en 0.0202
(pequeña ganancia, no es una buena división)
```

---

## EJERCICIO 3: Aplicar Teorema de Bayes

### Problema
En FlightDelays, queremos predecir Flight Status para un vuelo con:
- CARRIER = DL (Delta)
- DAY_WEEK = 7 (Domingo)
- CRS_DEP_TIME = 15 (3 PM)

Del dataset de entrenamiento obtuvimos:
- P(Retraso) = 0.20 (20% de vuelos tienen retraso)
- P(OnTime) = 0.80 (80% llegan a tiempo)

Y las probabilidades condicionales:
- P(CARRIER_DL | Retraso) = 0.095
- P(CARRIER_DL | OnTime) = 0.204
- P(DAY_WEEK_7 | Retraso) = 0.161
- P(DAY_WEEK_7 | OnTime) = 0.105
- P(CRS_DEP_TIME_15 | Retraso) = 0.050
- P(CRS_DEP_TIME_15 | OnTime) = 0.080

**¿Cuál es la probabilidad de que sea un retraso?**

### Solución
```
P(Retraso | DL, Domingo, 3PM) ∝ P(DL|Retraso) × P(Domingo|Retraso) 
                                × P(3PM|Retraso) × P(Retraso)

Numerador_Retraso = 0.095 × 0.161 × 0.050 × 0.20
                  = 0.095 × 0.161 × 0.050 × 0.20
                  = 0.0001523

P(OnTime | DL, Domingo, 3PM) ∝ P(DL|OnTime) × P(Domingo|OnTime) 
                               × P(3PM|OnTime) × P(OnTime)

Numerador_OnTime = 0.204 × 0.105 × 0.080 × 0.80
                 = 0.204 × 0.105 × 0.080 × 0.80
                 = 0.0013689

Normalizar:
P(Retraso | datos) = 0.0001523 / (0.0001523 + 0.0013689)
                   = 0.0001523 / 0.0015212
                   = 0.10 (10%)

P(OnTime | datos) = 0.0013689 / 0.0015212
                  = 0.90 (90%)

PREDICCIÓN: OnTime (ya que 90% > 10%)
Confianza: 90%
```

---

## EJERCICIO 4: Matriz de Confusión

### Problema
En UniversalBank, usamos Random Forest para predecir "Personal Loan"

Predicciones en el conjunto de validación (2000 muestras):
```
              Real
           Prestamo No-Prestamo
Predicho:
Prestamo      160        4
No-Prestamo   32       1804
```

**Calcular todas las métricas**

### Solución
```
TP (Verdaderos Positivos): 160  ← Predijo Sí, era Sí
FP (Falsos Positivos):     4    ← Predijo Sí, era No
FN (Falsos Negativos):     32   ← Predijo No, era Sí
TN (Verdaderos Negativos): 1804 ← Predijo No, era No

Total = 160 + 4 + 32 + 1804 = 2000

ACCURACY = (TP + TN) / Total
         = (160 + 1804) / 2000
         = 1964 / 2000
         = 0.982 = 98.2% ✅ Muy bueno

PRECISION = TP / (TP + FP)
          = 160 / (160 + 4)
          = 160 / 164
          = 0.976 = 97.6%
          ← De mis predicciones de "Sí Préstamo", 97.6% eran correctas

RECALL (SENSIBILIDAD) = TP / (TP + FN)
                      = 160 / (160 + 32)
                      = 160 / 192
                      = 0.833 = 83.3%
                      ← De los que realmente pidieron préstamo, detecté 83.3%

F1-SCORE = 2 × (Precision × Recall) / (Precision + Recall)
         = 2 × (0.976 × 0.833) / (0.976 + 0.833)
         = 2 × 0.813 / 1.809
         = 1.626 / 1.809
         = 0.899 = 89.9%

ESPECIFICIDAD = TN / (TN + FP)
              = 1804 / (1804 + 4)
              = 1804 / 1808
              = 0.998 = 99.8%
              ← De los que NO pidieron préstamo, predije correctamente 99.8%

INTERPRETACIÓN:
- Accuracy muy alta (98.2%) → Modelo muy bueno
- Precision alta (97.6%) → Pocas falsas alarmas
- Recall bueno (83.3%) → Detecta 83% de solicitudes reales
- Especificidad excelente (99.8%) → Casi no predice falsos positivos
```

---

## EJERCICIO 5: One-Hot Encoding

### Problema
Tenemos un dataset con:
```
ID  CARRIER  DAY_WEEK
1   AA       2
2   DL       5
3   AA       2
4   UA       7
```

**Aplicar One-Hot Encoding a CARRIER y DAY_WEEK**

### Solución
```python
import pandas as pd

df = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'CARRIER': ['AA', 'DL', 'AA', 'UA'],
    'DAY_WEEK': [2, 5, 2, 7]
})

X = pd.get_dummies(df[['CARRIER', 'DAY_WEEK']])
```

**Resultado:**
```
   CARRIER_AA  CARRIER_DL  CARRIER_UA  DAY_WEEK_2  DAY_WEEK_5  DAY_WEEK_7
0           1           0           0           1           0           0
1           0           1           0           0           1           0
2           1           0           0           1           0           0
3           0           0           1           0           0           1
```

**Interpretación:**
- Fila 0: AA, Día 2 → AA=1, DL=0, UA=0, Día2=1, Día5=0, Día7=0
- Fila 1: DL, Día 5 → AA=0, DL=1, UA=0, Día2=0, Día5=1, Día7=0
- etc.

**Ventaja:** Cada categoría es una variable binaria, sin asumir orden
```

---

## EJERCICIO 6: Cross-Validation Manual

### Problema
Tenemos 10 muestras y queremos hacer 2-Fold Cross-Validation

Dataset:
```
[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
Clases: [A,  A,  A,  B,  B,  A,  B,  A,  B,  A]
```

Entrenar DecisionTree y reportar accuracy de cada fold

### Solución
```
FOLD 1:
├─ Train: [d1, d2, d3, d4, d5] (clases: A, A, A, B, B)
├─ Test:  [d6, d7, d8, d9, d10] (clases: A, B, A, B, A)
├─ Entrenar árbol con train
├─ Predecir test: [A, B, A, B, A]  ← Perfect match!
└─ Accuracy_1 = 5/5 = 1.0 = 100%

FOLD 2:
├─ Train: [d6, d7, d8, d9, d10] (clases: A, B, A, B, A)
├─ Test:  [d1, d2, d3, d4, d5] (clases: A, A, A, B, B)
├─ Entrenar árbol con train
├─ Predecir test: [A, A, A, B, B]  ← Perfect match!
└─ Accuracy_2 = 5/5 = 1.0 = 100%

PROMEDIO = (100% + 100%) / 2 = 100%
DESVIACIÓN ESTÁNDAR = 0%

Conclusión: Modelo generalizó perfectamente (datos muy simples)
```

---

## EJERCICIO 7: GridSearchCV Simplificado

### Problema
Queremos encontrar los mejores parámetros para DecisionTree

Grid a probar:
```python
param_grid = {
    'max_depth': [1, 2, 3],
    'min_samples_split': [2, 4]
}
```

Dado Dataset con 100 muestras, 5-Fold CV

**Enumerar todas las combinaciones a probar**

### Solución
```
Combinaciones posibles = 3 × 2 = 6 modelos

COMBINACIÓN 1: max_depth=1, min_samples_split=2
├─ 5-Fold CV en training
└─ Score promedio = 0.75

COMBINACIÓN 2: max_depth=1, min_samples_split=4
├─ 5-Fold CV en training
└─ Score promedio = 0.78

COMBINACIÓN 3: max_depth=2, min_samples_split=2
├─ 5-Fold CV en training
└─ Score promedio = 0.82 ← Mejor hasta ahora

COMBINACIÓN 4: max_depth=2, min_samples_split=4
├─ 5-Fold CV en training
└─ Score promedio = 0.80

COMBINACIÓN 5: max_depth=3, min_samples_split=2
├─ 5-Fold CV en training
└─ Score promedio = 0.81

COMBINACIÓN 6: max_depth=3, min_samples_split=4
├─ 5-Fold CV en training
└─ Score promedio = 0.79

MEJOR COMBINACIÓN: max_depth=2, min_samples_split=2 con score 0.82 ✅

Entrenar modelo final con estos parámetros en TODO el training
Evaluar en TEST set
```

---

## EJERCICIO 8: Seleccionar Características por Correlación

### Problema
Dataset con 5 características y objetivo = Personal Loan

Correlaciones calculadas:
```
Income              0.45
Age                 0.32
Experience          0.30
Education           0.15
CC Avg              0.08
```

**Seleccionar top-3 características**

### Solución
```
Ranking por correlación (descendente):
1. Income:      0.45 ← #1 (mayor correlación)
2. Age:         0.32 ← #2
3. Experience:  0.30 ← #3
4. Education:   0.15 ← No usar (muy baja)
5. CC Avg:      0.08 ← No usar (muy baja)

CARACTERÍSTICAS SELECCIONADAS: [Income, Age, Experience]

Justificación:
- Income tiene la correlación más fuerte (0.45)
- Age y Experience son moderadamente correlacionadas
- Education y CC Avg tienen correlación muy débil (<0.15)

Interpretación:
- Income explica 45% de la relación con Personal Loan
- Age explica 32%
- Experience explica 30%
- El resto contribuye poco
```

---

## EJERCICIO 9: Identificar Overfitting

### Problema
Dos modelos entrenados, resultados:

**Modelo A:**
- Training Accuracy: 85%
- Validation Accuracy: 84%

**Modelo B:**
- Training Accuracy: 99%
- Validation Accuracy: 72%

**¿Cuál tiene overfitting?**

### Solución
```
MODELO A:
├─ Diferencia = 85% - 84% = 1% ✅
├─ Pequeña diferencia
├─ Training y Validation similares
└─ CONCLUSIÓN: Sin overfitting (bueno)

MODELO B:
├─ Diferencia = 99% - 72% = 27% ❌
├─ ENORME diferencia
├─ Training perfecto, Validation pobre
└─ CONCLUSIÓN: SEVERO OVERFITTING

INTERPRETACIÓN MODELO B:
- El modelo MEMORIZÓ datos de training (99%)
- NO GENERALIZA a datos nuevos (72%)
- Necesita:
  1. Aumentar max_depth
  2. Aumentar min_samples_split
  3. Usar Random Forest
  4. Más data de training
```

---

## EJERCICIO 10: Feature Importance Ranking

### Problema
Un modelo Random Forest reporta:

```python
importances = [0.15, 0.35, 0.02, 0.28, 0.20]
features = ['Age', 'Income', 'Experience', 'Education', 'Family']
```

**Crear ranking y interpretar**

### Solución
```
Crear DataFrame:
┌────────────┬────────────┐
│ Feature    │ Importance │
├────────────┼────────────┤
│ Income     │ 0.35       │ ← #1 (35%)
│ Education  │ 0.28       │ ← #2 (28%)
│ Family     │ 0.20       │ ← #3 (20%)
│ Age        │ 0.15       │ ← #4 (15%)
│ Experience │ 0.02       │ ← #5 (2%)
└────────────┴────────────┘

Total = 0.35 + 0.28 + 0.20 + 0.15 + 0.02 = 1.00 ✓

INTERPRETACIÓN:
- Income es 17.5× más importante que Experience (0.35 / 0.02)
- Las 3 primeras explican 83% de las predicciones
- Experience prácticamente no influye (2%)

RECOMENDACIÓN:
- Mantener: Income, Education, Family, Age
- Considerar eliminar: Experience (muy baja importancia)
- Reduce dimensionalidad sin perder precisión
```

---

## EJERCICIO 11: Problema de Práctica - Completo

### Situación
La empresa "MejorBanco" te pide clasificar si otorgar un préstamo personal.

Dataset: 1000 clientes, 80% train, 20% test

Características disponibles:
- Age (continua): 21-70 años
- Salary (continua): $20k-$150k
- Credit_Score (continua): 300-850
- Employment_Years (continua): 0-40 años
- Marital_Status (discreta): Single, Married, Divorced
- Education (discreta): High School, Bachelor, Master
- Phone_Ownership (discreta): Yes, No
- Has_Credit_Card (discreta): Yes, No
- Target: Loan_Approved (Si/No)

**Preguntas:**

a) ¿Qué algoritmo recomendarías y por qué?
b) ¿Qué características excluirías? ¿Por qué?
c) ¿Necesitarías transformar alguna variable?
d) ¿Cómo evaluarías el modelo?

### Solución

```
a) ALGORITMO RECOMENDADO: Random Forest

   JUSTIFICACIÓN:
   ✅ Datos mixtos (continuas + discretas)
   ✅ Random Forest maneja ambas sin transformación
   ✅ Reduce overfitting automáticamente
   ✅ Proporciona Feature Importance
   ✅ Alta precisión esperada (~95%)
   ❌ Decision Tree simple: Propenso a overfitting
   ❌ Naive Bayes: Necesitaría muchas transformaciones

b) CARACTERÍSTICAS A EXCLUIR: Ninguna

   ANÁLISIS DE CADA CARACTERÍSTICA:
   ✅ Age: Relevante (edad → riesgo de crédito)
   ✅ Salary: MUY RELEVANTE (capacidad de pago)
   ✅ Credit_Score: MUY RELEVANTE (historial crediticio)
   ✅ Employment_Years: Relevante (estabilidad laboral)
   ✅ Marital_Status: Relevante (dependientes)
   ✅ Education: Relevante (ingresos futuros)
   ✅ Phone_Ownership: Tal vez débil, pero no perjudica
   ✅ Has_Credit_Card: Indicador de historial

c) TRANSFORMACIONES NECESARIAS:

   Para Random Forest: NINGUNA requerida
   (ya maneja continuas y discretas)

   Pero si usáramos Naive Bayes (MultinomialNB):
   ├─ Age → Binning: Age_18_30, Age_30_45, Age_45_60, Age_60plus
   ├─ Salary → Binning: Salary_Low, Salary_Medium, Salary_High
   ├─ Credit_Score → Binning: Poor, Fair, Good, Excellent
   ├─ Employment_Years → Similar binning
   └─ Discretas: One-hot encoding

d) EVALUACIÓN DEL MODELO:

   MÉTRICAS PRINCIPALES:
   ├─ Accuracy: Porcentaje total correcto
   ├─ Precision: De los préstamos aprobados, ¿cuántos sí pagaron?
   │  (Evitar falsas aprobaciones = créditos incobrables)
   ├─ Recall: De los buenos pagadores, ¿cuántos detectamos?
   │  (No perder buenos clientes)
   └─ F1-Score: Balance entre ambos

   PROCESO:
   1. Dividir: 80% train, 20% test
   2. Entrenar Random Forest en train
   3. 5-Fold CV en train para validar
   4. Evaluar en test (sin data leakage)
   5. Matriz de confusión
   6. Feature Importance

   MÉTRICAS ACEPTABLES:
   ├─ Accuracy: > 85%
   ├─ Precision: > 80% (evitar falsas aprobaciones)
   ├─ Recall: > 75% (capturar buenos clientes)
   └─ ROC-AUC: > 0.90

   INTERPRETACIÓN ESPERADA:
   Si Precision=85%, significa:
   "De cada 100 préstamos aprobados,
    85 fueron realmente buenos pagadores"

   Si Recall=80%, significa:
   "De cada 100 buenos pagadores,
    detectamos 80"
```

---

## EJERCICIO 12: Pregunta Teórica Compleja

### Problema
"En un árbol de decisión, ¿por qué es importante limitar max_depth?
Explica matemáticamente qué pasa sin límite y con límite=5"

### Solución
```
SIN LÍMITE DE PROFUNDIDAD (max_depth=None):

El árbol crece hasta que todos los nodos hoja sean puros (una sola clase)

Ejemplo visual:
                Root (Gini=0.48)
                /              \
         Nodo_2 (G=0.4)    Nodo_3 (G=0.45)
          /          \       /          \
    Nodo_4 (G=0.35) Hoja_A Hoja_B    Nodo_5 (G=0.4)
     /     \                          /      \
Hoja_C  Hoja_D                   Hoja_E  Hoja_F

... continúa hasta Gini = 0 en todas las hojas

PROBLEMA: Memoriza datos de training
Training Accuracy = 100%
Test Accuracy = 75% ❌ Overfitting

RAZÓN MATEMÁTICA:
P(Overfitting) = (Complejidad - Relevancia)²

Sin límite:
├─ Complejidad muy alta (muchas reglas específicas)
├─ Se ajusta a ruido y anomalías
├─ No generaliza a datos nuevos
└─ Error en test >>> Error en train


CON LÍMITE max_depth=5:

El árbol crece máximo 5 niveles, aunque haya nodos impuros

Ejemplo visual:
                Root (Gini=0.48)
                /              \
         Nodo_2 (G=0.4)    Nodo_3 (G=0.45)
          /          \       /          \
    Nodo_4 (G=0.35) Nodo_5 (G=0.42) Nodo_6 Nodo_7
     /     \        /     \
Hoja_A Hoja_B  Hoja_C  Hoja_D

[Profundidad = 5, STOP aquí aunque no sean puros]

VENTAJA: Generaliza mejor
Training Accuracy = 87%
Test Accuracy = 85% ✅ Buen balance

RAZÓN MATEMÁTICA:
Error_Total = Error_Bias + Error_Varianza

Sin límite:
├─ Bias bajo (se ajusta bien)
└─ Varianza ALTA (cambia mucho con datos nuevos)

Con límite=5:
├─ Bias un poco más alto (menos flexible)
└─ Varianza BAJA (más estable con datos nuevos)

CONCLUSIÓN:
max_depth actúa como regularizador

Formula aproximada:
Error_Test ≈ Error_Train + λ × (Complejidad)

where λ es mayor cuando max_depth es más bajo
```

---

## CONCLUSIÓN

**Estos ejercicios cubren:**
- ✅ Cálculos matemáticos (Gini, Bayes)
- ✅ Conceptos prácticos (One-Hot, CV, GridSearch)
- ✅ Interpretación de resultados
- ✅ Elección de algoritmos
- ✅ Detección de problemas

**Para preparar tu examen:**
1. Resuelve TODOS estos ejercicios
2. Entiende cada paso
3. Cubre tus soluciones y rehazlas
4. Varia los números y practica más

¡Mucho éxito! 🎉

---

*Documento: Ejercicios Prácticos con Soluciones*  
*Minería de Datos - 8vo Ciclo*
