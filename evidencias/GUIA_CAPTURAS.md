# Evidencias reales de MLflow (pendientes de generar)

1. En Colab ejecutar `fuentes/entrena.ipynb` con MLflow instalado. Verificar que aparezca `mlruns/`.
2. En entorno local, ejecutar `mlflow ui --backend-store-uri ./mlruns` desde la raíz y abrir http://127.0.0.1:5000. Si usas Colab, descarga la carpeta `mlruns` para abrirla localmente o usa un túnel autenticado; no expongas una UI pública sin protección.
3. Captura A: lista de runs `Regresion_logistica`, `Random_forest` y `comparacion_global` con estado FINISHED.
4. Captura B: pestaña Metrics con accuracy, precision, recall, f1, roc_auc, cv_auc_media y cv_auc_sd.
5. Captura C: pestaña Parameters con hiperparámetros seleccionados, `cv_folds=5` y `dataset_version`.
6. Captura D: pestaña Artifacts mostrando modelo serializado, matriz de confusión, predicciones y curvas ROC.
7. Guardar capturas PNG en `evidencias/` y registrarlas con commit. **No presentar imágenes de ejemplo como resultados reales.**
