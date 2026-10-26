# Diccionario de datos v1

Fuente: **generación sintética programática**, 600 filas. No contiene información de clientes reales. Objetivo generado probabilísticamente a partir de variables predictoras; por ello el benchmark refleja la simulación, no la realidad.

| Campo | Tipo | Descripción |
|---|---|---|
| id_cliente | texto | Identificador artificial SYNxxxxx; no se usa como predictor |
| edad | entero | Edad (años); se invalidan valores fuera de 18-100 |
| antiguedad_anios | entero | Antigüedad simulada |
| saldo_mxn | decimal | Saldo simulado; negativos pasan a nulo |
| num_productos | entero | Número de productos simulados |
| usa_app | binario | Uso de app móvil (1 sí) |
| reclamaciones_12m | entero | Reclamaciones simuladas último año |
| meses_inactividad | entero | Meses de inactividad simulados |
| region | categoría | Norte, Centro, Sur |
| segmento | categoría | Masivo, Preferente, PyME |
| score_crediticio | numérico | Score simulado 350-850 |
| abandono | binario | Etiqueta objetivo simulada: 1 abandono, 0 permanencia |

Se introdujeron faltantes deliberados en saldo, score y región, y anomalías en edad y saldo. No se infiere causalidad de las correlaciones.
