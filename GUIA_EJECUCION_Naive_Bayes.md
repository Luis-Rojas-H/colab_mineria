# 🚀 GUÍA DE EJECUCIÓN: Multinomial Naive Bayes FlightDelays

## 📌 Nombre del Archivo Creado
**`08_Multinomial_Naive_Bayes_FlightDelays.ipynb`**

Este nombre es descriptivo porque:
- **08** = Número del tema (Naive Bayes es típicamente tema 8)
- **Multinomial_Naive_Bayes** = Algoritmo usado
- **FlightDelays** = Dataset
- **.ipynb** = Formato Jupyter Notebook

---

## 🔧 INSTALACIÓN DE DEPENDENCIAS

Antes de ejecutar, instala estas librerías:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn dmba jupyter
```

---

## 📂 ESTRUCTURA DEL NOTEBOOK

El notebook tiene **9 secciones principales**:

### 1️⃣ **Instalación de Dependencias**
   - Instala `dmba` automáticamente

### 2️⃣ **Importaciones de Librerías**
   - pandas, numpy, sklearn, matplotlib, dmba

### 3️⃣ **Carga y Exploración de Datos**
   - Lee `FlightDelays.csv`
   - Muestra forma y tipos de datos

### 4️⃣ **Preparación de Datos**
   **5 pasos clave:**
   - ✅ Convertir a categorías
   - ✅ Crear bins horarios
   - ✅ Seleccionar predictores
   - ✅ One-hot encoding
   - ✅ División train/validation

### 5️⃣ **Entrenamiento del Modelo**
   - Crea clasificador MultinomialNB(alpha=0.01)
   - Lo ajusta con datos de training

### 6️⃣ **Predicciones**
   - Predicción de clases (`predict`)
   - Predicción de probabilidades (`predict_proba`)

### 7️⃣ **Análisis de Probabilidades Condicionales**
   - Para cada predictor, muestra tabla de probabilidades
   - Responde: ¿Qué predictor influye más en retrasos?

### 8️⃣ **Predicción Detallada para Caso Específico**
   - Busca vuelos Delta que salen domingo a las 10 AM
   - Muestra sus predicciones exactas

### 9️⃣ **Evaluación del Modelo**
   - Calcula Accuracy
   - Muestra matriz de confusión

---

## 🎯 ¿QUÉ DEBES ANALIZAR?

### **Pregunta 1: ¿Cuál es el predictor más importante?**
→ Mira las tablas de probabilidades condicionales
→ ¿Qué predictor tiene mayores diferencias entre "ontime" y "delayed"?

**Ejemplo (hipotético):**
```
DAY_WEEK:
  Lunes-viernes: ~65% ontime
  Fin de semana: ~55% ontime ← Mayor diferencia

CRS_DEP_TIME:
  Mañana (6-9): ~70% ontime
  Tarde (15-18): ~60% ontime ← También influye
```

### **Pregunta 2: ¿Por qué es importante train_test_split?**
→ Para evitar **overfitting**
→ Validamos en datos que el modelo **nunca vio**

### **Pregunta 3: ¿Por qué one-hot encoding?**
→ MultinomialNB necesita **números**, no textos
→ Convierte `CARRIER=['DL', 'AA']` en columnas `CARRIER_DL=[0,1,0...]`

### **Pregunta 4: ¿Qué es alpha=0.01?**
→ Laplace Smoothing
→ Previene probabilidades cero
→ Si una aerolínea nunca aparece en training, no dice P=0

### **Pregunta 5: Diferencia predict() vs predict_proba()**
```python
predict() → ['ontime', 'delayed', 'ontime', ...]
           (solo la clase ganadora)

predict_proba() → [[0.75, 0.25],
                    [0.30, 0.70],
                    [0.92, 0.08], ...]
                 (TODAS las probabilidades)
```

---

## 📊 MÉTRICAS QUE VERÁS

### **Accuracy**
```
Correcto / Total = correctas / todas
Rango: 0 a 1 (0% a 100%)
```

### **Matriz de Confusión**
```
Real ontime:
  ✅ Predicho ontime (True Positive)
  ❌ Predicho delayed (False Negative)

Real delayed:
  ❌ Predicho ontime (False Positive)
  ✅ Predicho delayed (True Negative)
```

---

## 🎓 COMPARATIVA: TUS DOS PREGUNTAS DE EXAMEN

### **Pregunta 1: SparsePCA (datos discretos)**
- **Tipo:** Reducción dimensional
- **Datos:** Necesita continuos
- **Resultado:** Componentes principales
- **Problema:** No funciona con WEATHER_R, TRAF_CON_R (discretos)

### **Pregunta 2: Multinomial NB (FlightDelays)**
- **Tipo:** Clasificación
- **Datos:** Funciona con categóricos/discretos
- **Resultado:** Predicción de clase + probabilidades
- **Solución:** Ideal para este problema

---

## 💻 CÓMO EJECUTAR

### **Opción 1: En Jupyter Notebook (Recomendado)**
```bash
jupyter notebook 08_Multinomial_Naive_Bayes_FlightDelays.ipynb
```

### **Opción 2: En Google Colab**
1. Copia el contenido del `.ipynb`
2. Pega en nueva celda de Colab
3. Ejecuta células secuencialmente

### **Opción 3: Convertir a Python script**
```bash
jupyter nbconvert --to script 08_Multinomial_Naive_Bayes_FlightDelays.ipynb
python 08_Multinomial_Naive_Bayes_FlightDelays.py
```

---

## ⚠️ POSIBLES ERRORES Y SOLUCIONES

| Error | Causa | Solución |
|-------|-------|----------|
| `FileNotFoundError: FlightDelays.csv` | Archivo no en carpeta actual | Usa ruta completa o `cd` a directorio correcto |
| `ModuleNotFoundError: pandas` | No instalado | `pip install pandas` |
| `ValueError: no columns to parse` | CSV corrupto | Verifica separadores (`,` vs `;`) |
| `Error en predict_proba` | Predictor con valores no vistos | No afecta el flujo, sigue adelante |

---

## 📝 RESUMEN EJECUTIVO

**El notebook completo:**
1. ✅ Carga datos de retrasos de vuelos
2. ✅ Prepara variables categóricas
3. ✅ Entrena modelo Naive Bayes
4. ✅ Hace predicciones
5. ✅ Analiza probabilidades
6. ✅ Evalúa rendimiento

**Lo que aprendes:**
- Cómo aplicar Naive Bayes a problemas reales
- Importancia de preparación de datos
- Cómo interpretar resultados de clasificación
- Diferencia entre train y validation

---

## 🎯 PARA TU EXAMEN

**Prepárate para explicar:**

1. **"¿Cuál es la diferencia entre SparsePCA y Naive Bayes?"**
   → SparsePCA es para exploración, Naive Bayes es para predicción

2. **"¿Por qué dividimos en train/test?"**
   → Para evaluar en datos nuevos (sin overfitting)

3. **"¿Qué hace get_dummies()?"**
   → Convierte categorías en variables binarias (one-hot encoding)

4. **"¿Qué es Laplace Smoothing?"**
   → Evita probabilidades cero (parámetro alpha)

5. **"¿Cómo interpretamos P(ontime|CARRIER_DL)?"**
   → Probabilidad de que un vuelo sea puntual DADO que es Delta Airlines

---

**¡Ahora sí estás listo para tu examen!** 🚀

