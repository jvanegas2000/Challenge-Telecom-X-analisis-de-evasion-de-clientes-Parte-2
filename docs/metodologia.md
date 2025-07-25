# **Metodología: Predicción de Cancelación de Clientes (Churn) - Telecom X**

## **1. Introducción**

Esta metodología describe el enfoque sistemático utilizado para desarrollar un modelo predictivo de cancelación de clientes (churn) para Telecom X. El proyecto sigue las mejores prácticas de ciencia de datos, implementando un pipeline completo desde la exploración inicial hasta la evaluación final del modelo.

### **1.1 Objetivos Metodológicos**
- 🎯 **Desarrollo de modelo predictivo** con alta precisión (F1-Score > 80%)
- 🔍 **Identificación de factores críticos** que influyen en la cancelación
- 🛠️ **Creación de pipeline reproducible** para implementación en producción
- 📊 **Generación de insights accionables** para estrategias de retención

### **1.2 Enfoque General**
La metodología sigue el framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining) adaptado para el contexto específico de predicción de churn:

```
Entendimiento del Negocio → Entendimiento de Datos → Preparación de Datos → 
Modelado → Evaluación → Implementación
```

---

## **2. Descripción del Dataset**

### **2.1 Características del Dataset**
- **Fuente**: Datos históricos de clientes de Telecom X
- **Tamaño**: 7,043 registros × 21 variables
- **Tipo**: Dataset etiquetado para aprendizaje supervisado
- **Variable objetivo**: `Churn` (0 = No canceló, 1 = Canceló)

### **2.2 Tipos de Variables**

#### **Variables Numéricas:**
- `tenure`: Tiempo de permanencia del cliente (meses)
- `MonthlyCharges`: Cargo mensual ($)
- `TotalCharges`: Cargo total acumulado ($)
- `DailyCharges`: Cargo diario ($)

#### **Variables Categóricas:**
- `gender`: Género del cliente
- `SeniorCitizen`: Cliente de tercera edad (0/1)
- `Partner`: Tiene pareja (Yes/No)
- `Dependents`: Tiene dependientes (Yes/No)
- `PhoneService`: Servicio telefónico (Yes/No)
- `MultipleLines`: Múltiples líneas telefónicas
- `InternetService`: Tipo de servicio de internet
- `OnlineSecurity`: Seguridad online (Yes/No/No internet service)
- `OnlineBackup`: Respaldo online
- `DeviceProtection`: Protección de dispositivos
- `TechSupport`: Soporte técnico
- `StreamingTV`: TV en streaming
- `StreamingMovies`: Películas en streaming
- `Contract`: Tipo de contrato (Month-to-month/One year/Two year)
- `PaperlessBilling`: Facturación sin papel (Yes/No)
- `PaymentMethod`: Método de pago

---

## **3. Análisis Exploratorio de Datos (EDA)**

### **3.1 Análisis Univariado**

#### **Distribución de la Variable Objetivo**
```python
# Análisis de proporción de churn
churn_counts = df['Churn'].value_counts()
churn_proportion = df['Churn'].value_counts(normalize=True)

# Visualización
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Churn')
plt.title('Distribución de Churn')
```

**Hallazgos clave:**
- **Desbalance de clases**: ~26.5% churn vs 73.5% no-churn
- **Necesidad de técnicas de balanceo** para evitar sesgo hacia la clase mayoritaria

#### **Variables Numéricas**
```python
# Análisis de distribuciones
numeric_vars = ['tenure', 'MonthlyCharges', 'TotalCharges', 'DailyCharges']
for var in numeric_vars:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df[var], kde=True)
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df[var])
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='Churn', y=var)
```

**Observaciones:**
- `tenure`: Distribución sesgada hacia valores bajos
- `MonthlyCharges`: Distribución aproximadamente normal
- `TotalCharges`: Fuertemente correlacionado con tenure
- Presencia de outliers en todas las variables numéricas

### **3.2 Análisis Bivariado**

#### **Correlación entre Variables Numéricas**
```python
# Matriz de correlación
corr_matrix = df[numeric_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
```

**Hallazgos de correlación:**
- `DailyCharges` ↔ `MonthlyCharges`: r = 1.0 (colinealidad perfecta)
- `TotalCharges` ↔ `tenure`: r = 0.86 (alta correlación)
- Necesario análisis de multicolinealidad

#### **Variables Categóricas vs Churn**
```python
# Análisis de asociación
categorical_vars = df.select_dtypes(include=['object']).columns
for var in categorical_vars:
    if var != 'Churn':
        crosstab = pd.crosstab(df[var], df['Churn'], normalize='index')
        crosstab.plot(kind='bar', figsize=(10, 6))
```

**Patrones identificados:**
- **Contract**: Clientes month-to-month tienen mayor churn
- **PaymentMethod**: Electronic check asociado con mayor cancelación
- **Internet Services**: Fiber optic muestra tasas elevadas de churn

---

## **4. Preprocesamiento de Datos**

### **4.1 Limpieza de Datos**

#### **Tratamiento de Valores Faltantes**
```python
# Identificación de valores faltantes
missing_values = df.isnull().sum()
print("Valores faltantes por columna:")
print(missing_values[missing_values > 0])

# Estrategia: Eliminación de filas con NaN
df_clean = df.dropna()
print(f"Registros eliminados: {len(df) - len(df_clean)}")
```

**Justificación:**
- Pocos valores faltantes (<1% del dataset)
- Eliminación no afecta significativamente el tamaño del dataset
- Preserva la integridad de las relaciones entre variables

#### **Conversión de Tipos de Datos**
```python
# TotalCharges: string → numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Churn: Yes/No → 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

### **4.2 Codificación de Variables Categóricas**

#### **One-Hot Encoding**
```python
# Variables categóricas para codificar
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('Churn')  # Excluir variable objetivo

# Aplicar One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
```

**Ventajas del método:**
- ✅ No asume orden en variables categóricas
- ✅ Evita sesgo de codificación ordinal arbitraria
- ✅ Compatible con todos los algoritmos de ML

### **4.3 División del Dataset**

```python
from sklearn.model_selection import train_test_split

# Separar características y variable objetivo
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Configuración:**
- **Proporción**: 80% entrenamiento / 20% prueba
- **Estratificación**: Mantiene proporción de clases en ambos conjuntos
- **Random state**: 42 (reproducibilidad)

---

## **5. Balanceo de Clases**

### **5.1 Análisis del Desbalance**
```python
# Distribución antes del balanceo
print("Distribución original:")
print(y_train.value_counts(normalize=True))
```

**Problema identificado:**
- Clase 0 (No churn): ~73.5%
- Clase 1 (Churn): ~26.5%
- **Ratio**: 2.8:1 (desbalance significativo)

### **5.2 Aplicación de SMOTE**

```python
from imblearn.over_sampling import SMOTE

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Distribución después de SMOTE:")
print(pd.Series(y_train_balanced).value_counts(normalize=True))
```

**Características de SMOTE:**
- 🎯 **Genera muestras sintéticas** para la clase minoritaria
- 🔄 **Preserva la estructura** de los datos originales
- ⚖️ **Equilibra las clases** (50%-50%)
- 🚫 **No duplica registros** existentes

**Justificación de la elección:**
- Superior a oversampling simple (evita overfitting)
- Mejor que undersampling (no pierde información)
- Específicamente diseñado para problemas de clasificación

---

## **6. Normalización de Datos**

### **6.1 Necesidad de Normalización**

```python
# Análisis de escalas
print("Estadísticas descriptivas:")
print(X_train_balanced.describe())
```

**Observaciones:**
- Variables con diferentes escalas (tenure: 0-72, TotalCharges: 0-8684)
- Algoritmos sensibles a escala (SVM, KNN) requieren normalización

### **6.2 Aplicación de StandardScaler**

```python
from sklearn.preprocessing import StandardScaler

# Ajustar scaler solo en datos de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Convertir de vuelta a DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_balanced.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

**Ventajas de StandardScaler:**
- ✅ **Media = 0, Desviación estándar = 1**
- ✅ **Mantiene la forma** de la distribución
- ✅ **Robusto** para la mayoría de algoritmos ML
- ✅ **Previene data leakage** (fit solo en entrenamiento)

---

## **7. Selección y Entrenamiento de Modelos**

### **7.1 Algoritmos Seleccionados**

La selección se basó en:
- **Diversidad de enfoques**: Ensemble, SVM, Instance-based, Linear
- **Robustez ante desbalance**: Capacidad de manejar clases balanceadas
- **Interpretabilidad**: Importancia de variables para insights de negocio

#### **Modelos Implementados:**

1. **Random Forest**
   ```python
   from sklearn.ensemble import RandomForestClassifier

   rf_params = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 15, 20, None],
       'min_samples_split': [2, 5],
       'min_samples_leaf': [1, 2]
   }
   ```

2. **Gradient Boosting**
   ```python
   from sklearn.ensemble import GradientBoostingClassifier

   gb_params = {
       'n_estimators': [100, 200],
       'learning_rate': [0.05, 0.1, 0.15],
       'max_depth': [3, 5, 7]
   }
   ```

3. **Support Vector Machine**
   ```python
   from sklearn.svm import SVC

   svm_params = {
       'C': [0.1, 1, 10],
       'kernel': ['rbf', 'linear'],
       'gamma': ['scale', 'auto']
   }
   ```

4. **K-Nearest Neighbors**
   ```python
   from sklearn.neighbors import KNeighborsClassifier

   knn_params = {
       'n_neighbors': [3, 5, 7, 9],
       'weights': ['uniform', 'distance'],
       'metric': ['euclidean', 'manhattan']
   }
   ```

5. **Logistic Regression**
   ```python
   from sklearn.linear_model import LogisticRegression

   lr_params = {
       'C': [0.01, 0.1, 1, 10],
       'penalty': ['l1', 'l2'],
       'solver': ['liblinear', 'saga']
   }
   ```

### **7.2 Optimización de Hiperparámetros**

#### **GridSearchCV Configuration**
```python
from sklearn.model_selection import GridSearchCV

# Configuración común
grid_search_params = {
    'cv': 5,                    # 5-fold cross-validation
    'scoring': 'f1',            # Métrica de optimización
    'n_jobs': -1,               # Paralelización
    'verbose': 1                # Progress tracking
}

# Ejemplo para Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    **grid_search_params
)

rf_grid.fit(X_train_scaled, y_train_balanced)
```

**Justificación de F1-Score como métrica:**
- 🎯 **Balanceada**: Considera tanto precision como recall
- 📊 **Apropiada para clases balanceadas**: Post-SMOTE
- 🔍 **Sensible a falsos positivos y negativos**: Crítico en churn prediction

### **7.3 Validación Cruzada**

#### **5-Fold Cross-Validation**
```python
# Proceso de validación cruzada
fold_1: 80% train, 20% validation
fold_2: 80% train, 20% validation  
fold_3: 80% train, 20% validation
fold_4: 80% train, 20% validation
fold_5: 80% train, 20% validation

# Métrica final = promedio de 5 folds
```

**Beneficios:**
- ✅ **Uso eficiente** de datos de entrenamiento
- ✅ **Estimación robusta** del rendimiento
- ✅ **Detección de overfitting** mediante comparación train vs validation
- ✅ **Selección confiable** de hiperparámetros

---

## **8. Evaluación de Modelos**

### **8.1 Métricas de Evaluación**

#### **Métricas Implementadas:**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }
    return metrics
```

#### **Interpretación de Métricas:**

1. **Accuracy**: Proporción de predicciones correctas
   - Útil cuando las clases están balanceadas

2. **Precision**: TP / (TP + FP)
   - "De los clientes que predijimos como churn, ¿cuántos realmente cancelaron?"

3. **Recall (Sensitivity)**: TP / (TP + FN)
   - "De todos los clientes que cancelaron, ¿a cuántos detectamos?"

4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - **Métrica principal**: Balance entre precision y recall

5. **AUC**: Área bajo la curva ROC
   - Capacidad de discriminación entre clases

### **8.2 Matriz de Confusión**

```python
# Análisis detallado de errores
cm = confusion_matrix(y_test, y_pred)

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Actual')
plt.xlabel('Predicho')
```

**Análisis de errores:**
- **True Positives (TP)**: Churn correctamente identificado
- **False Positives (FP)**: Cliente retenido clasificado como churn
- **False Negatives (FN)**: Cliente con churn no detectado ⚠️ **Crítico**
- **True Negatives (TN)**: Cliente retenido correctamente clasificado

### **8.3 Curvas ROC**

```python
from sklearn.metrics import roc_curve

# Generar curva ROC para cada modelo
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC - Comparación de Modelos')
plt.legend()
```

---

## **9. Análisis de Multicolinealidad**

### **9.1 Variance Inflation Factor (VIF)**

#### **Cálculo de VIF**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Análisis inicial
vif_results = calculate_vif(X_train_scaled)
print("Variables con VIF > 10:")
print(vif_results[vif_results['VIF'] > 10])
```

#### **Interpretación de VIF:**
- `VIF = 1`: No correlación con otras variables
- `VIF < 5`: Multicolinealidad baja
- `5 ≤ VIF < 10`: Multicolinealidad moderada
- `VIF ≥ 10`: **Multicolinealidad severa** ⚠️

### **9.2 Matriz de Correlación**

```python
# Identificar correlaciones altas
corr_matrix = X_train_scaled.corr()
upper_triangle = corr_matrix.where(
    np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
)

# Pares con correlación > 0.8
high_corr_pairs = []
for column in upper_triangle.columns:
    for index in upper_triangle.index:
        if abs(upper_triangle.loc[index, column]) > 0.8:
            high_corr_pairs.append((index, column, upper_triangle.loc[index, column]))
```

### **9.3 Eliminación de Variables Redundantes**

#### **Criterios de Eliminación:**
1. **VIF > 10**: Multicolinealidad severa
2. **Correlación > 0.8**: Alta correlación lineal
3. **Importancia baja**: Según Random Forest feature importance

#### **Variables Eliminadas:**
```python
# Variables identificadas para eliminación
variables_to_remove = [
    'PhoneService',                    # VIF alto + baja importancia
    'TotalCharges',                    # Alta correlación con tenure
    'DailyCharges',                    # Correlación perfecta con MonthlyCharges
    'InternetService_Fiber optic'      # VIF alto
]

# Dataset optimizado
X_train_optimized = X_train_scaled.drop(variables_to_remove, axis=1)
X_test_optimized = X_test_scaled.drop(variables_to_remove, axis=1)
```

### **9.4 Comparación de Modelos**

```python
# Entrenar modelo original vs optimizado
rf_original = RandomForestClassifier(**best_params)
rf_optimized = RandomForestClassifier(**best_params)

rf_original.fit(X_train_scaled, y_train_balanced)
rf_optimized.fit(X_train_optimized, y_train_balanced)

# Comparar rendimiento
original_metrics = evaluate_model(rf_original, X_test_scaled, y_test)
optimized_metrics = evaluate_model(rf_optimized, X_test_optimized, y_test)
```

---

## **10. Interpretabilidad del Modelo**

### **10.1 Importancia de Variables**

```python
# Feature importance del mejor modelo
feature_importance = pd.DataFrame({
    'Variable': X_train_optimized.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualización
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), y='Variable', x='Importance')
plt.title('Top 15 Variables Más Importantes')
```

### **10.2 Análisis de Importancia**

#### **Top 5 Variables Críticas:**
1. **tenure** (11.7%): Tiempo de permanencia
2. **TotalCharges** (11.4%): Valor total facturado
3. **PaymentMethod_Electronic check** (9.2%): Método de pago
4. **MonthlyCharges** (8.7%): Cargo mensual
5. **OnlineSecurity** (8.1%): Servicio de seguridad

**Insights de negocio:**
- 🕒 **Clientes nuevos** tienen mayor riesgo
- 💰 **Facturación** es factor crítico
- 💳 **Método de pago** influye significativamente
- 🔒 **Servicios adicionales** impactan retención

---

## **11. Validación Final**

### **11.1 Prueba en Conjunto de Test**

```python
# Evaluación final del mejor modelo
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train_optimized, y_train_balanced)

# Predicciones en conjunto de prueba
y_pred_final = final_model.predict(X_test_optimized)
y_proba_final = final_model.predict_proba(X_test_optimized)[:, 1]

# Métricas finales
final_metrics = evaluate_model(final_model, X_test_optimized, y_test)
```

### **11.2 Análisis de Residuos**

```python
# Análisis de casos mal clasificados
misclassified = X_test_optimized[y_test != y_pred_final]
print(f"Casos mal clasificados: {len(misclassified)}")

# Patrones en errores
error_analysis = pd.DataFrame({
    'Actual': y_test[y_test != y_pred_final],
    'Predicted': y_pred_final[y_test != y_pred_final],
    'Probability': y_proba_final[y_test != y_pred_final]
})
```

---

## **12. Consideraciones Metodológicas**

### **12.1 Fortalezas de la Metodología**

- ✅ **Pipeline robusto**: Desde EDA hasta evaluación final
- ✅ **Validación cruzada**: Estimación confiable del rendimiento
- ✅ **Múltiples modelos**: Comparación exhaustiva de algoritmos
- ✅ **Optimización sistemática**: GridSearchCV para hiperparámetros
- ✅ **Análisis de multicolinealidad**: Modelo interpretable y estable
- ✅ **Métricas apropiadas**: F1-Score para datasets balanceados

### **12.2 Limitaciones Identificadas**

- ⚠️ **Datos estáticos**: No considera evolución temporal
- ⚠️ **Variables limitadas**: Dataset podría enriquecerse con más features
- ⚠️ **SMOTE**: Genera datos sintéticos, no reales
- ⚠️ **Threshold fijo**: Podría optimizarse para casos de uso específicos

### **12.3 Recomendaciones para Implementación**

#### **Monitoreo del Modelo**
```python
# Métricas a monitorear en producción
monitoring_metrics = [
    'model_accuracy_drift',
    'feature_distribution_shift', 
    'prediction_distribution_change',
    'business_metrics_impact'
]
```

#### **Reentrenamiento**
- **Frecuencia**: Cada 3-6 meses
- **Triggers**: Degradación del performance > 5%
- **Datos nuevos**: Incorporar feedback de intervenciones

---

## **13. Reproducibilidad**

### **13.1 Seeds y Estados Aleatorios**
```python
# Configuración para reproducibilidad
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# En cada algoritmo
RandomForestClassifier(random_state=RANDOM_SEED)
train_test_split(random_state=RANDOM_SEED)
SMOTE(random_state=RANDOM_SEED)
```

### **13.2 Versionado de Código**
- 📝 **Jupyter Notebook**: Código completo documentado
- 🔧 **Requirements.txt**: Versiones específicas de librerías
- 📊 **Datos**: Dataset original preservado
- 📋 **Resultados**: Métricas y visualizaciones guardadas

---

**Esta metodología garantiza un enfoque científico y reproducible para el desarrollo de modelos predictivos de churn, siguiendo las mejores prácticas de la industria y proporcionando resultados confiables para la toma de decisiones de negocio.**
