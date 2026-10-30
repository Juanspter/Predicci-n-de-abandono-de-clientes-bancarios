# Actividad 5 | Predicción de abandono bancario (datos sintéticos)

**Objetivo:** comparar regresión logística y random forest mediante búsqueda de hiperparámetros, validación cruzada estratificada de 5 folds y seguimiento de experimentos con MLflow. Caso académico: **no apto para decisiones bancarias reales**.

## Datos y decisiones
- Origen: **generación sintética programática**, semilla 2026; **800 registros**; clase abandono positiva 12.5%. No se usan datos de clientes reales. La BBDD sintética fue creada con ayuda de chatgpt utilizando un prompt que dirigiera el resultado hacia un entregable  con información demográfica, perfil bancario del usuario/cliente. Eso llevaría a que fuera reproducible en un ambiente académico controlado y sobre todo, habiéndose probado y asegurado su usabilidad, llevarlo a entornos reales y privados.
- Original: `datos/datos_ini/clientes_sinteticos_v1.csv`; limpio: `datos/datos_limp/clientes_limpios_v1.csv`.
- `cliente_id` es identificador y se excluye de entrenamiento. `abandono` es objetivo binario (1 = abandona).
- Variables: edad (años), antiguedad_anios, num_productos, saldo_mxn, cliente_activo, transacciones_digitales_mes, reclamaciones_12m, region, segmento.
- Valores faltantes: **se conservan** en CSV limpio y se imputan **dentro de cada fold** para evitar fuga de datos. Mediana para numéricas, moda para categóricas; OneHotEncoder para categorías, StandardScaler para numéricas.
- Separación: 75% entrenamiento / 25% prueba estratificada, semilla 42. GridSearchCV optimiza ROC-AUC con 5 folds; el holdout solo se evalúa al final.

## Ejecución
```bash
pip install -r requirements.txt
python fuentes/train.py --raiz .
mlflow ui --backend-store-uri ./mlruns
```
Abre `http://127.0.0.1:5000` en tu computadora para consultar los experimentos. El notebook `fuentes/entrena.ipynb` permite subir el ZIP de este proyecto a Colab y ejecutar el mismo pipeline.

**Importante:** en este entorno se ejecutó y validó el entrenamiento **sin MLflow**, porque no estaba disponible para instalar. El registro en MLflow está implementado, pero **requiere ejecutarse en Colab o localmente** para generar evidencia auténtica. La carpeta `mlruns/` no se incluye en el ZIP y está ignorada por Git.

## Resultados reales de la ejecución local sin MLflow
| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC media |
|---|---:|---:|---:|---:|---:|---:|
| Regresion_logistica | 0.875 | 0.000 | 0.000 | 0.000 | 0.771 | 0.732 |
| Random_forest | 0.875 | 0.000 | 0.000 | 0.000 | 0.748 | 0.708 |

**Hallazgo crítico:** ambos modelos clasificaron todo el conjunto de prueba como no abandono al umbral 0.5 (recall positivo = 0). Aunque accuracy es alta por desbalance, **ninguno está listo para uso operativo**. ROC-AUC indica capacidad de ordenamiento, pero falta ajustar umbral y evaluar costos de errores.

## Evidencia MLflow
Después de ejecutar con MLflow, capturar: (1) experimento y runs; (2) pestaña Metrics de cada run; (3) pestaña Parameters; (4) Artifacts (modelo, predicciones, matriz de confusión y ROC). Ver `evidencias/GUIA_CAPTURAS.md`. No se adjuntan capturas simuladas.

## Archivos
- `fuentes/datos_prep.py`: limpieza reproducible.
- `fuentes/train.py`: GridSearchCV, métricas, artefactos y MLflow.
- `fuentes/entrena.ipynb`: notebook para Colab.
- `resultados/`: resultados **obtenidos realmente** en esta ejecución, sin registro MLflow.
- `CHANGELOG.md`: cambios y versiones.
- `reportes/reporte_tecnico.pdf`: informe y limitaciones.

## Control de versiones
Crear un repositorio vacío `Actividad5` en GitHub, descomprimir el ZIP y subir su contenido, o usar Git:
```bash
git init
git add .
git commit -m "Actividad 5: baseline y experimentos"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/Actividad5.git
git push -u origin main
```
Disclaimer para reproducción del caso: No subas secretos, datos reales de clientes ni el servidor MLflow abierto a Internet.
