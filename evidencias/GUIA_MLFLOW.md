# Evidencia de MLflow: cómo capturarla

**No se incluyen capturas simuladas como si fueran reales.** Para generar evidencia auténtica:

1. En Colab instala dependencias (`pip install -r requirements.txt`) y ejecuta `python fuentes/train.py`.
2. Comprueba que aparezca `MLflow: registrado en mlruns/` y que exista el directorio `mlruns`.
3. En un equipo local, ejecuta `mlflow ui --backend-store-uri ./mlruns --port 5000` y abre http://127.0.0.1:5000.
4. Captura **Experiments** mostrando `Actividad5_Churn_Sintetico` y dos runs.
5. Captura la comparación de runs mostrando accuracy, precision, recall, f1, roc_auc y cv_f1_mean.
6. Abre cada run y captura **Parameters**, **Metrics**, **Artifacts** (matriz, CSV grid y modelo).
7. Guarda imágenes como `evidencias/mlflow_01_experimentos.png`, `mlflow_02_comparacion.png`, `mlflow_03_artefactos.png` y actualiza el reporte.

En Google Colab la UI local no es accesible directamente desde tu navegador sin un túnel; también puedes descargar `mlruns/` y abrirlo en tu equipo. No publiques tokens ni URLs de túneles privadas.
