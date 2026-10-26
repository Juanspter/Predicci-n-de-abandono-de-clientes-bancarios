"""Limpieza determinista; imputación y escalado se aprenden SOLO en cada fold."""
import pandas as pd
from pathlib import Path

def limpiar(df):
    d = df.copy()
    d = d.drop_duplicates(subset=["cliente_id"])
    for c in ["region", "segmento"]:
        d[c] = d[c].astype("string").str.strip().str.title().replace({"<Na>": pd.NA})
    for c in ["edad", "antiguedad_anios", "num_productos", "saldo_mxn", "cliente_activo", "transacciones_digitales_mes", "reclamaciones_12m", "abandono"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["abandono"].isin([0,1]) & d["edad"].between(18,100)]
    return d.reset_index(drop=True)

def preparar_archivo(raiz):
    raiz = Path(raiz)
    d = limpiar(pd.read_csv(raiz / "datos/datos_ini/clientes_sinteticos_v1.csv"))
    salida = raiz / "datos/datos_limp/clientes_limpios_v1.csv"
    salida.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(salida,index=False)
    return salida
