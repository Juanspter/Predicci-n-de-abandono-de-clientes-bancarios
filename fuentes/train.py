"""Entrenamiento de churn sintético con GridSearchCV y MLflow.
Uso: python fuentes/train.py --raiz .
"""
import argparse, json, hashlib, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from datos_prep import preparar_archivo

NUM = ["edad", "antiguedad_anios", "num_productos", "saldo_mxn", "cliente_activo", "transacciones_digitales_mes", "reclamaciones_12m"]
CAT = ["region", "segmento"]

def preprocesador():
    return ColumnTransformer([
        ("numericas", Pipeline([("imputar",SimpleImputer(strategy="median")), ("escalar",StandardScaler())]), NUM),
        ("categoricas", Pipeline([("imputar",SimpleImputer(strategy="most_frequent")), ("onehot",OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])

def ejecutar(raiz, usar_mlflow=True):
    raiz=Path(raiz).resolve(); salida=raiz/"resultados"; salida.mkdir(exist_ok=True)
    preparar_archivo(raiz)
    df=pd.read_csv(raiz/"datos/datos_limp/clientes_limpios_v1.csv")
    X=df[NUM+CAT]; y=df["abandono"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    modelos={
        "Regresion_logistica":(LogisticRegression(max_iter=1000,random_state=42),{"clasificador__C":[.1,1,10],"clasificador__class_weight":[None,"balanced"]}),
        "Random_forest":(RandomForestClassifier(random_state=42,n_jobs=1),{"clasificador__n_estimators":[100,200],"clasificador__max_depth":[4,8,None],"clasificador__min_samples_leaf":[1,5]})
    }
    mlflow=None
    if usar_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as e:
            raise RuntimeError("Instala MLflow con: pip install -r requirements.txt") from e
        mlflow.set_tracking_uri((raiz/"mlruns").as_uri())
        mlflow.set_experiment("Actividad5_Churn_Sintetico")
    rows=[]; folds={}; curvas={}
    for nombre,(est,grid) in modelos.items():
        pipe=Pipeline([("preprocesador",preprocesador()),("clasificador",est)])
        busqueda=GridSearchCV(pipe,grid,cv=cv,scoring="roc_auc",n_jobs=-1,refit=True)
        busqueda.fit(Xtr,ytr)
        pred=busqueda.predict(Xte); proba=busqueda.predict_proba(Xte)[:,1]
        scores=cross_val_score(busqueda.best_estimator_,Xtr,ytr,cv=cv,scoring="roc_auc",n_jobs=-1)
        folds[nombre]=scores.tolist()
        row={"modelo":nombre,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,proba),"cv_auc_media":float(scores.mean()),"cv_auc_sd":float(scores.std(ddof=1)),"n_train":len(ytr),"n_test":len(yte),"mejores_parametros":json.dumps(busqueda.best_params_,ensure_ascii=False)}
        rows.append(row); curvas[nombre]=roc_curve(yte,proba)[:2]
        cm=confusion_matrix(yte,pred,labels=[0,1]); np.savetxt(salida/f"confusion_{nombre}.csv",cm,fmt="%d",delimiter=",")
        pd.DataFrame({"real":yte.to_numpy(),"predicho":pred,"probabilidad_abandono":proba}).to_csv(salida/f"predicciones_{nombre}.csv",index=False)
        if mlflow:
            with mlflow.start_run(run_name=nombre):
                mlflow.log_params(busqueda.best_params_)
                mlflow.log_param("dataset_version","v1_sintetico")
                mlflow.log_param("cv_folds",5)
                mlflow.log_param("random_state",42)
                mlflow.log_metrics({k:float(row[k]) for k in ["accuracy","precision","recall","f1","roc_auc","cv_auc_media","cv_auc_sd"]})
                mlflow.log_artifact(str(salida/f"confusion_{nombre}.csv"))
                mlflow.log_artifact(str(salida/f"predicciones_{nombre}.csv"))
                mlflow.sklearn.log_model(busqueda.best_estimator_,artifact_path="modelo")
    tabla=pd.DataFrame(rows); tabla.to_csv(salida/"comparacion_modelos.csv",index=False)
    estad,pvalor=ttest_rel(folds["Regresion_logistica"],folds["Random_forest"]); (salida/"comparacion_estadistica.json").write_text(json.dumps({"metodo":"t pareada exploratoria sobre 5 folds compartidos; no prueba confirmatoria", "t":float(estad),"p_valor":float(pvalor),"fold_auc":folds},indent=2))
    plt.figure(figsize=(7,5))
    for nombre,(fpr,tpr) in curvas.items():plt.plot(fpr,tpr,label=nombre)
    plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('Tasa de falsos positivos');plt.ylabel('Recall');plt.title('Curvas ROC - holdout');plt.legend();plt.tight_layout();plt.savefig(salida/'curvas_roc.png',dpi=160);plt.close()
    if mlflow:
        with mlflow.start_run(run_name="comparacion_global"):
            mlflow.log_artifact(str(salida/'comparacion_modelos.csv'))
            mlflow.log_artifact(str(salida/'comparacion_estadistica.json'))
            mlflow.log_artifact(str(salida/'curvas_roc.png'))
    print(tabla.to_string(index=False));print('Resultados:',salida)
    return tabla

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raiz',default='.');p.add_argument('--sin-mlflow',action='store_true');args=p.parse_args()
    ejecutar(args.raiz,usar_mlflow=not args.sin_mlflow)
