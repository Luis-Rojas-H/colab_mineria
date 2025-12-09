# 📚 FEEDBACK: ÁRBOLES DE DECISIÓN Y RANDOM FOREST

## 📌 NOMBRE DEL ARCHIVO

**`09_Decision_Trees_RandomForest_Classification.ipynb`**

**Por qué este nombre:**
- `09` = Tema 9 (Árboles de Decisión y Ensambles)
- `Decision_Trees_RandomForest` = Algoritmos usados
- `Classification` = Tipo de problema
- `.ipynb` = Formato ejecutable

---

## 🎯 ¿QUÉ ES ESTE NOTEBOOK?

Implementa **3 enfoques distintos** de clasificación:

```
1. Árboles de Decisión Simples (depth=1)
2. Árboles de Decisión Completos (sin restricciones)
3. Árboles Optimizados (GridSearchCV)
4. Random Forest (Ensamble)
```

---

## 📂 ESTRUCTURA DEL NOTEBOOK (14 Secciones)

### **PARTE 1: ÁRBOLES DE DECISIÓN**

```
1. Importaciones
   ↓
2. Árbol Simple (max_depth=1, Mowers dataset)
   ├─ Árbol muy simple para entender concepto
   └─ Fácil de visualizar
   
3. Árbol Completo (sin restricciones, UniversalBank)
   ├─ Árbol sin límites
   └─ Propenso a overfitting
   
4. Evaluación del árbol completo
   ├─ Compara train vs validation
   └─ Observa overfitting
   
5. Cross-Validation (5-fold)
   └─ Validación más robusta
   
6. Árbol Podado (Pruned tree)
   ├─ Con restricciones (max_depth, min_samples_split)
   └─ Más pequeño y generalizable
   
7. Evaluación del árbol podado
   └─ Mejor balance train/validation
   
8. GridSearchCV - Búsqueda inicial
   ├─ Búsqueda gruesa de parámetros
   └─ Identifica zona promisoria
   
9. GridSearchCV - Búsqueda refinada
   ├─ Búsqueda fina alrededor de buenos parámetros
   └─ Encuentra óptimos locales
   
10. Evaluación del árbol optimizado
    └─ El mejor árbol que encontramos
```

### **PARTE 2: RANDOM FOREST**

```
11. Random Forest Classifier
    ├─ 500 árboles
    ├─ Cada uno entrenado con muestra aleatoria
    └─ Predicción = promedio de todos
    
12. Feature Importance
    ├─ Qué variables son más importantes
    ├─ Gráfico de barras
    └─ Barras de error (desviación estándar)
    
13. Evaluación Random Forest
    └─ Accuracy en validation
    
14. Resumen Comparativo
    └─ Comparación de los 3 modelos
```

---

## 🔑 CONCEPTOS CLAVE

### **Árbol de Decisión (Decision Tree)**

```
¿Qué es?
├─ Modelo que divide datos en regiones
├─ Usa división binaria (if/else)
└─ Crea reglas legibles

Ventajas:
├─ Fácil de entender y explicar
├─ No requiere normalización
└─ Maneja categorías bien

Desventajas:
├─ Propenso a overfitting
├─ Inestable (pequeños cambios = árbol diferente)
└─ Sesgado a variables de alta cardinalidad

Parámetros importantes:
├─ max_depth: Profundidad máxima (controla tamaño)
├─ min_samples_split: Mínimo para dividir (controla crecimiento)
└─ min_impurity_decrease: Mejora mínima (previene divisiones débiles)
```

### **Random Forest (Ensamble)**

```
¿Qué es?
├─ Conjunto de árboles de decisión
├─ Cada árbol se entrena con bootstrap (muestra con reemplazo)
├─ Cada árbol usa características aleatorias
└─ Predicción = promedio de todos los árboles

Ventajas:
├─ Reduce overfitting significativamente
├─ Más robusto que árbol individual
├─ Mejor generalización
├─ Proporciona Feature Importance

Desventajas:
├─ Menos interpretable (no ves el árbol final)
├─ Más lento de entrenar (pero predice rápido)
└─ Usa más memoria

Parámetro clave:
└─ n_estimators: Número de árboles (típicamente 100-1000)
```

### **GridSearchCV**

```
¿Qué es?
├─ Búsqueda exhaustiva de hiperparámetros
├─ Prueba todas las combinaciones posibles
└─ Usa cross-validation para evaluar

Proceso:
1. Define grid de parámetros a probar
2. Para cada combinación:
   ├─ Entrena modelo
   ├─ Evalúa con cross-validation
   └─ Guarda score
3. Retorna mejor combinación

Ejemplo:
├─ Búsqueda 1 (gruesa): 100 combinaciones
├─ Analiza resultados
└─ Búsqueda 2 (fina): 500 combinaciones en zona promisoria
```

---

## 📊 ¿QUÉ DEBES ANALIZAR?

### **Sección 2-4: Árbol Simple y Completo**

```
Pregunta: ¿Por qué el árbol completo tiene accuracy=1.0 en training?
Respuesta: Overfitting - memorizó el dataset de training

Pregunta: ¿Por qué baja el accuracy en validation?
Respuesta: No generalizó bien - no vio esos casos antes
```

### **Sección 5: Cross-Validation**

```
Pregunta: ¿Son similares los 5 folds?
Respuesta: Si son parecidos → modelo estable
           Si varían mucho → modelo inestable

Pregunta: ¿Es mejor el árbol completo que el promedio de CV?
Respuesta: No necesariamente - CV es más confiable
```

### **Sección 6-7: Árbol Podado**

```
Pregunta: ¿Mejora vs árbol completo?
Respuesta: Training: baja (es más simple)
           Validation: sube (generaliza mejor)
           
Pregunta: ¿Cuál es el balance ideal?
Respuesta: Cuando train ≈ validation (sin gran diferencia)
```

### **Sección 8-10: GridSearchCV y Optimización**

```
Pregunta: ¿Cuántas combinaciones se prueban?
Búsqueda 1: 4 × 5 × 5 = 100 combinaciones
Búsqueda 2: 14 × 12 × 3 = 504 combinaciones

Pregunta: ¿Por qué dos búsquedas?
Respuesta: 1) Exploración gruesa → zona promisoria
           2) Exploración fina → óptimo local
```

### **Sección 12: Feature Importance**

```
Pregunta: ¿Qué significan las barras?
Respuesta: Altura = importancia
           Línea de error = variabilidad entre árboles

Pregunta: ¿Cuáles son las variables más importantes?
Respuesta: Las más largas (mayores importancias)

Pregunta: ¿Por qué Random Forest puede decir importancia?
Respuesta: Cada árbol contribuye diferente
           Promediamos importancias = variable importance del forest
```

---

## 🎯 PARA TU EXAMEN

### **Pregunta 1: ¿Diferencia entre Decision Tree y Random Forest?**

```
Decision Tree:
├─ Un solo árbol
├─ Interpre table
└─ Propenso a overfitting

Random Forest:
├─ Muchos árboles (típicamente 500)
├─ Menos interpretable pero más robusto
└─ Mejor generalización
```

### **Pregunta 2: ¿Por qué GridSearchCV?**

```
Sin GridSearchCV:
└─ Adivinas parámetros (trial and error)

Con GridSearchCV:
├─ Prueba combinaciones sistemáticamente
├─ Usa cross-validation (más confiable)
└─ Encuentra buenos parámetros automáticamente
```

### **Pregunta 3: ¿Cómo sé si hay overfitting?**

```
Indicadores:
├─ train_accuracy >> valid_accuracy
├─ train error muy bajo, valid error alto
└─ Árbol muy profundo

Soluciones:
├─ Reducir max_depth
├─ Aumentar min_samples_split
├─ Usar Random Forest en lugar de árbol simple
└─ Más datos (si es posible)
```

### **Pregunta 4: ¿Qué es Feature Importance?**

```
Concepto:
├─ Mide cuán importante es cada variable
├─ Se calcula cómo contribuye a reducir impureza
└─ Valores entre 0 y 1 (suma a 1.0)

Interpretación:
├─ Importancia > 0.2 = muy importante
├─ Importancia 0.05-0.2 = moderadamente importante
└─ Importancia < 0.05 = poco importante
```

---

## 📝 COMPARACIÓN CON OTRAS TÉCNICAS

```
┌──────────────────┬─────────────────┬──────────────────┬─────────────────┐
│ Característica   │ Decision Tree   │ Random Forest    │ Naive Bayes     │
├──────────────────┼─────────────────┼──────────────────┼─────────────────┤
│ Interpretable    │ ✅ Muy sí       │ ⚠️ Poco          │ ✅ Muy sí       │
│ Velocidad train  │ ✅ Rápido       │ ⚠️ Medio         │ ✅ Muy rápido   │
│ Velocidad pred   │ ✅ Muy rápido   │ ⚠️ Lento (500árboles)             │
│ Overfitting      │ ❌ Alto riesgo  │ ✅ Muy resistente│ ⚠️ Medio        │
│ Datos continuos  │ ✅ Sí           │ ✅ Sí            │ ⚠️ Requiere bins│
│ Datos discretos  │ ✅ Sí           │ ✅ Sí            │ ✅ Ideal        │
│ Escalado needed  │ ❌ No           │ ❌ No            │ ⚠️ Recomendado  │
└──────────────────┴─────────────────┴──────────────────┴─────────────────┘
```

---

## ✅ CHECKLIST PRE-EXAMEN

- [ ] Entiendo la diferencia entre Decision Tree y Random Forest
- [ ] Puedo explicar qué es overfitting
- [ ] Entiendo cómo funciona GridSearchCV
- [ ] Sé qué significan los parámetros (max_depth, min_samples_split)
- [ ] Puedo interpretar una matriz de confusión
- [ ] Entiendo Feature Importance
- [ ] Puedo leer un árbol de decisión (cuando se plotea)
- [ ] Conozco ventajas y desventajas de cada modelo
- [ ] Sé cuándo usar Decision Tree vs Random Forest vs Naive Bayes

---

## 🚀 CÓMO EJECUTAR

```bash
# Requisitos (si no los tienes aún):
pip install scikit-learn pandas numpy matplotlib seaborn dmba jupyter

# Ejecutar:
cd "c:\Users\USUARIO\Documents\GitHub\colab_mineria"
jupyter notebook 09_Decision_Trees_RandomForest_Classification.ipynb
```

---

## 🎓 CONCEPTOS CLAVE A RECORDAR

```
1. Decision Tree = Un árbol que divide recursivamente
2. Random Forest = Múltiples árboles para mayor robustez
3. Overfitting = Alta precisión en train, baja en validation
4. Underfitting = Baja precisión en ambos (modelo muy simple)
5. GridSearchCV = Búsqueda automática de mejores hiperparámetros
6. Feature Importance = Qué variables influyen más en predicciones
7. Cross-Validation = Evaluar modelo sin depender de una división específica
8. Bootstrap = Muestra con reemplazo (base de Random Forest)
```

---

**¡Éxito en tu examen!** 🏆
