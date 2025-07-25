# Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2

# 📊 Customer Churn Prediction - Telecom X

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un proyecto completo de Machine Learning para predecir la cancelación de clientes (churn) en una empresa de telecomunicaciones, utilizando técnicas avanzadas de análisis de datos y modelado predictivo.

## 📋 Tabla de Contenidos

- [🎯 Descripción del Proyecto](#-descripción-del-proyecto)
- [✨ Características Principales](#-características-principales)
- [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [🚀 Instalación y Configuración](#-instalación-y-configuración)
- [📊 Uso del Proyecto](#-uso-del-proyecto)
- [🏆 Resultados Principales](#-resultados-principales)
- [📈 Visualizaciones](#-visualizaciones)
- [🔍 Metodología](#-metodología)
- [💡 Insights de Negocio](#-insights-de-negocio)
- [🤝 Contribuciones](#-contribuciones)
- [📄 Licencia](#-licencia)
- [📞 Contacto](#-contacto)

## 🎯 Descripción del Proyecto

Este proyecto desarrolla un sistema de predicción de churn (cancelación de clientes) para Telecom X, una empresa de telecomunicaciones. El objetivo es identificar clientes con alta probabilidad de cancelar sus servicios, permitiendo implementar estrategias proactivas de retención.

### Objetivos Principales:
- 🔮 **Predicción Precisa**: Modelo con 84.9% de F1-Score
- 📊 **Análisis Profundo**: Identificación de factores de riesgo
- 🎯 **Estrategias Accionables**: Recomendaciones específicas de retención
- 🔧 **Pipeline Robusto**: Código reutilizable para producción

## ✨ Características Principales

- ✅ **Análisis Exploratorio Completo** con visualizaciones interactivas
- ✅ **Preprocesamiento Avanzado** (SMOTE, StandardScaler, One-hot Encoding)
- ✅ **Múltiples Algoritmos ML** (Random Forest, SVM, Gradient Boosting, etc.)
- ✅ **Optimización de Hiperparámetros** con GridSearchCV
- ✅ **Análisis de Multicolinealidad** (VIF, correlaciones)
- ✅ **Métricas Exhaustivas** (F1-Score, AUC, Precision, Recall)
- ✅ **Interpretabilidad del Modelo** con importancia de variables
- ✅ **Recomendaciones Estratégicas** basadas en resultados

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **Análisis de Datos** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-FF6B6B?style=flat) |
| **Visualización** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) |
| **Estadísticas** | ![Statsmodels](https://img.shields.io/badge/Statsmodels-4B8BBE?style=flat) |
| **Entorno** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=google-colab&logoColor=white) |

## 📁 Estructura del Proyecto

```
📦 Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2/
├── 📄 README.md
├── 📓 TelecomX_parte2_latam_Juan_Carlos_Vanegas_Molina.ipynb
├── 📊 data/
│   └── df_TelecomX_Data.csv
├── 📋 docs/
│   ├── conclusiones_generales.md
│   └── metodologia.md
└── 🔧 requirements.txt
```

## 🚀 Instalación y Configuración

### Prerequisitos
- Python 3.8 o superior
- Jupyter Notebook o Google Colab

### Instalación Local
```bash
# Clonar el repositorio
git clone https://github.com/jvanegas2000/Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2.git
cd Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter Notebook
jupyter notebook TelecomX_parte2_latam_Juan_Carlos_Vanegas_Molina.ipynb
```

### Uso en Google Colab
```python
# Subir el archivo df_TelecomX_Data.csv a Colab
# Ejecutar todas las celdas del notebook secuencialmente
```

## 📊 Uso del Proyecto

### Ejecución Paso a Paso

1. **📁 Carga de Datos**: Importa el dataset de Telecom X
2. **🔍 Análisis Exploratorio**: Visualiza distribuciones y patrones
3. **🧹 Preprocesamiento**: Limpia y transforma los datos
4. **⚖️ Balanceo de Clases**: Aplica SMOTE para equilibrar el dataset
5. **🤖 Entrenamiento**: Entrena 5 modelos diferentes con GridSearchCV
6. **📊 Evaluación**: Compara modelos usando múltiples métricas
7. **🔍 Análisis de Multicolinealidad**: Optimiza el modelo final
8. **📋 Conclusiones**: Genera insights y recomendaciones

### Ejemplo de Uso Rápido
```python
# Cargar el notebook y ejecutar todas las celdas
# Los resultados se mostrarán automáticamente

# Para predicciones en nuevos datos:
# best_model.predict(new_data)
```

## 🏆 Resultados Principales

### 🥇 Mejor Modelo: Random Forest

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **F1-Score** | **0.8487** | Excelente balance precisión-recall |
| **AUC** | **0.9290** | Muy buena capacidad discriminativa |
| **Accuracy** | **0.8456** | 84.6% de predicciones correctas |
| **Precision** | **0.8318** | 83.2% de predicciones positivas correctas |
| **Recall** | **0.8664** | 86.6% de casos positivos detectados |

### 📊 Comparación de Modelos

```
🏆 Random Forest:     F1=0.8487  AUC=0.9290
🥈 Gradient Boosting: F1=0.8433  AUC=0.9320  
🥉 SVM:               F1=0.8258  AUC=0.9075
4️⃣ KNN:               F1=0.8177  AUC=0.8915
5️⃣ Logistic Reg.:    F1=0.8088  AUC=0.8960
```

### 🔍 Variables Más Importantes

| Ranking | Variable | Importancia | Acción Recomendada |
|---------|----------|-------------|-------------------|
| 🥇 | **Tenure** | 11.7% | Programas de fidelización |
| 🥈 | **TotalCharges** | 11.4% | Revisión de precios |
| 🥉 | **PaymentMethod** | 9.2% | Incentivos método de pago |
| 4️⃣ | **MonthlyCharges** | 8.7% | Ofertas personalizadas |
| 5️⃣ | **OnlineSecurity** | 8.1% | Promoción de servicios |

## 📈 Visualizaciones

El proyecto incluye múltiples visualizaciones:

- 📊 **Distribución de Variables**: Histogramas y boxplots
- 🔗 **Matriz de Correlación**: Heatmap de relaciones
- 📈 **Curvas ROC**: Comparación de modelos
- 🎯 **Matriz de Confusión**: Análisis de errores
- 📋 **Importancia de Variables**: Rankings y gráficos de barras
- 🔍 **Análisis de Multicolinealidad**: VIF y correlaciones

## 🔍 Metodología

### Preprocesamiento
- **Limpieza**: Eliminación de valores faltantes
- **Codificación**: One-hot encoding para variables categóricas
- **Balanceo**: SMOTE para equilibrar clases (50-50)
- **Normalización**: StandardScaler para variables numéricas
- **División**: 80% entrenamiento / 20% prueba

### Modelado
- **Algoritmos**: 5 modelos (RF, GB, SVM, KNN, LR)
- **Optimización**: GridSearchCV con 5-fold CV
- **Validación**: Conjunto de prueba independiente
- **Métricas**: F1-Score como métrica principal

### Análisis Avanzado
- **Multicolinealidad**: VIF > 10 y correlaciones > 0.8
- **Optimización**: Eliminación de 4 variables redundantes
- **Comparación**: Modelo original vs optimizado

## 💡 Insights de Negocio

### 🎯 Factores de Riesgo Identificados
- **Clientes nuevos** (< 12 meses) tienen 3x más riesgo
- **Pago electrónico** aumenta probabilidad de churn en 25%
- **Falta de servicios adicionales** correlaciona con cancelación

### 💰 Impacto Financiero Estimado
- **ROI proyectado**: 3:1 en el primer año
- **Reducción de churn**: 15-25% con intervención temprana
- **Ahorro por cliente retenido**: $200-500

### 🎲 Estrategias Recomendadas
1. **Programa de onboarding** para clientes nuevos
2. **Incentivos** para cambio de método de pago
3. **Ofertas personalizadas** de servicios adicionales
4. **Sistema de alertas** automático para riesgo alto

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. 🍴 Fork el proyecto
2. 🌿 Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push a la rama (`git push origin feature/AmazingFeature`)
5. 🔄 Abre un Pull Request

### Ideas para Contribuir
- 📊 Nuevas visualizaciones
- 🤖 Algoritmos adicionales
- 🔧 Optimizaciones de código
- 📚 Mejoras en documentación
- 🧪 Tests unitarios

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License - Se permite uso comercial y modificación
```

## 📞 Contacto

🔗 **Link del Proyecto**: [https://github.com/jvanegas2000/Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2](https://github.com/jvanegas2000/Challenge-Telecom-X-analisis-de-evasion-de-clientes-Parte-2)

### 🌐 Sígueme en:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](www.linkedin.com/in/juan-carlos-vanegas-molina)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/jvanegas2000)

---

⭐ **¡Dale una estrella si este proyecto te fue útil!** ⭐

---

**📊 Estadísticas del Proyecto**

![GitHub stars](https://img.shields.io/github/stars/tu-usuario/customer-churn-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/tu-usuario/customer-churn-prediction?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/tu-usuario/customer-churn-prediction?style=social)
