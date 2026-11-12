"""
NoiseGuard - Passo 5: Random Forest x baseline

O que o modelo faz
  Prevê o ESTADO SONORO DAQUI A 2 MINUTOS (0 = dentro do limite, 1 = próximo,
  2 = acima) usando o histórico recente medido pelo NodeMCU.

Baseline (regra simples)
  "O estado daqui a 2 minutos será igual ao estado de agora", ou seja, o
  limiar fixo de 65 dB aplicado à leitura atual (é o que o NodeMCU já faz).

Uso
  pip install pandas scikit-learn matplotlib joblib

  # 1) Treinar e avaliar (baixa os dados direto do ThingSpeak)
  python noiseguard_ml.py treinar --canal SEU_CHANNEL_ID
  #    ...ou usando um CSV baixado manualmente do canal
  python noiseguard_ml.py treinar --csv feeds.csv

  # 2) Previsão ao vivo, enviando o resultado para o canal do modelo
  #    (Windows)  set THINGSPEAK_ML_WRITE_KEY=SUA_WRITE_KEY_DO_CANAL_ML
  #    (Linux/Mac) export THINGSPEAK_ML_WRITE_KEY=SUA_WRITE_KEY_DO_CANAL_ML
  python noiseguard_ml.py aovivo --canal SEU_CHANNEL_ID

Se o canal do sensor for privado, acrescente --read-key SUA_READ_API_KEY.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# ---------------- Parâmetros ----------------
HORIZONTE = 6                                   # 6 entradas x 20 s = 2 minutos
JANELAS = {"1min": 3, "3min": 9, "5min": 15}    # nº de entradas de 20 s
GAP_MAX_S = 90                                  # buraco maior que isso separa "sessões"
LIMITE_DB, MARGEM_DB = 65.0, 5.0                # iguais aos do código do NodeMCU
CAMPOS = ["amp_media", "amp_max", "db_estimado", "estado_baseline", "rssi_wifi"]
NOMES_CLASSE = ["dentro", "próximo", "acima"]
ARQ_MODELO = "modelo_rf.joblib"
OBRIGATORIOS = ["created_at", "amp_media", "amp_max", "db_estimado", "estado_baseline"]


# ---------------- Carga e limpeza ----------------
def padronizar_csv(df):
    """Deixa as colunas com os nomes dos campos, seja qual for o cabeçalho do CSV."""
    meta = {"created_at", "entry_id", "latitude", "longitude", "elevation", "status"}
    campos = [c for c in df.columns if c not in meta]
    return df.rename(columns=dict(zip(campos, CAMPOS)))


def limpar(df, verbose=True):
    df = df.copy()
    for c in CAMPOS:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=OBRIGATORIOS)
    n_vazias = n0 - len(df)
    ok = (df["amp_max"].between(0, 1023)
          & df["db_estimado"].between(30, 120)
          & df["estado_baseline"].isin([0, 1, 2]))
    n_fora = int((~ok).sum())
    df = (df[ok].drop_duplicates("created_at")
          .sort_values("created_at").reset_index(drop=True))
    if verbose:
        print(f"Registros: {n0} lidos | {n_vazias} incompletos | "
              f"{n_fora} fora da faixa | {len(df)} válidos")
    return df


def carregar(args):
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.canal:
        url = f"https://api.thingspeak.com/channels/{args.canal}/feeds.csv?results=8000"
        if args.read_key:
            url += "&api_key=" + args.read_key
        df = pd.read_csv(url)
    else:
        sys.exit("Informe --canal ID ou --csv arquivo.csv")
    return limpar(padronizar_csv(df))


# ---------------- Características (features) ----------------
def montar_features(df):
    """Características calculadas só com o passado (sem olhar o futuro)."""
    gap = df["created_at"].diff().dt.total_seconds()
    seg = (gap.isna() | (gap > GAP_MAX_S)).cumsum()      # sessões contínuas
    db = df["db_estimado"]
    alto = (db >= LIMITE_DB - MARGEM_DB).astype(int)

    f = pd.DataFrame(index=df.index)
    f["db"] = db
    f["amp_media"] = df["amp_media"]
    f["amp_max"] = df["amp_max"]
    f["delta_db"] = db.groupby(seg).diff().fillna(0)
    for nome, n in JANELAS.items():
        r = db.groupby(seg).rolling(n, min_periods=n)
        f[f"db_media_{nome}"] = r.mean().reset_index(level=0, drop=True)
        f[f"db_max_{nome}"] = r.max().reset_index(level=0, drop=True)
        f[f"db_desvio_{nome}"] = r.std().reset_index(level=0, drop=True)
        f[f"picos_{nome}"] = (alto.groupby(seg).rolling(n, min_periods=n).sum()
                              .reset_index(level=0, drop=True))
    f["tendencia"] = f["db_media_1min"] - f["db_media_5min"]   # > 0 = subindo
    return f, seg


def montar_dataset(df):
    f, seg = montar_features(df)
    d = f.copy()
    d["alvo"] = df["estado_baseline"].groupby(seg).shift(-HORIZONTE)
    d["baseline"] = df["estado_baseline"]
    d["created_at"] = df["created_at"]
    d = d.dropna()
    d[["alvo", "baseline"]] = d[["alvo", "baseline"]].astype(int)
    return d


# ---------------- Treino e avaliação ----------------
def desenhar_matriz(ax, cm, titulo):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(titulo)
    ax.set_xticks(range(3), NOMES_CLASSE)
    ax.set_yticks(range(3), NOMES_CLASSE)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real (daqui a 2 min)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


def treinar(args):
    df = carregar(args)
    d = montar_dataset(df)
    cols = [c for c in d.columns if c not in ("alvo", "baseline", "created_at")]
    if len(d) < 100:
        sys.exit(f"Só {len(d)} linhas utilizáveis. Colete mais dados "
                 "(recomendo pelo menos 2 a 3 horas, ~500 entradas).")

    # Divisão temporal 80/20 (sem embaralhar). As últimas linhas do treino são
    # descartadas porque o alvo delas olha para dentro do período de teste.
    corte = int(len(d) * 0.8)
    treino, teste = d.iloc[:corte - HORIZONTE], d.iloc[corte:]

    dist = pd.DataFrame({"treino": treino["alvo"].value_counts(),
                         "teste": teste["alvo"].value_counts()})
    dist = dist.reindex([0, 1, 2]).fillna(0).astype(int)
    dist.index = NOMES_CLASSE
    print("\nDistribuição do alvo (estado daqui a 2 min):")
    print(dist.to_string())
    if treino["alvo"].nunique() < 2:
        sys.exit("O treino tem só uma classe. Varie o som durante TODA a coleta "
                 "(silêncio, conversa, música), não em blocos únicos.")
    if (teste["alvo"] == 2).sum() == 0:
        print("AVISO: o teste não tem nenhum caso 'acima'. As métricas dessa "
              "classe não valem. Colete mais dados com som alto espalhado no tempo.")

    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(treino[cols], treino["alvo"])

    y = teste["alvo"].to_numpy()
    p_rf = clf.predict(teste[cols])
    p_base = teste["baseline"].to_numpy()
    p_maior = np.full_like(y, treino["alvo"].mode()[0])
    labels = np.unique(y)

    def resumo(nome, p):
        return {"modelo": nome,
                "acurácia": accuracy_score(y, p),
                "F1 macro": f1_score(y, p, labels=labels, average="macro", zero_division=0),
                "precisão 'acima'": precision_score(y == 2, p == 2, zero_division=0),
                "recall 'acima'": recall_score(y == 2, p == 2, zero_division=0)}

    tabela = pd.DataFrame([resumo("Classe mais comum (referência)", p_maior),
                           resumo("Baseline (limiar fixo atual)", p_base),
                           resumo("Random Forest", p_rf)]).round(3)
    print(f"\nResultados no teste ({len(teste)} amostras, previsão a 2 min):")
    print(tabela.to_string(index=False))
    print("\nRelatório do Random Forest:")
    print(classification_report(y, p_rf, labels=[0, 1, 2], target_names=NOMES_CLASSE,
                                zero_division=0))

    # Gráficos para o relatório
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    desenhar_matriz(eixos[0], confusion_matrix(y, p_base, labels=[0, 1, 2]), "Baseline")
    desenhar_matriz(eixos[1], confusion_matrix(y, p_rf, labels=[0, 1, 2]), "Random Forest")
    fig.tight_layout()
    fig.savefig("matriz_confusao.png", dpi=150)

    imp = pd.Series(clf.feature_importances_, index=cols).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax2)
    ax2.set_title("Importância das características (Random Forest)")
    fig2.tight_layout()
    fig2.savefig("importancia_features.png", dpi=150)

    # Arquivos de saída
    prob = clf.predict_proba(teste[cols])
    p_acima = prob[:, list(clf.classes_).index(2)] if 2 in clf.classes_ else 0.0
    pd.DataFrame({"created_at": teste["created_at"].values, "real_2min": y,
                  "baseline": p_base, "random_forest": p_rf,
                  "prob_acima": np.round(p_acima, 3)}).to_csv(
        "resultados_teste.csv", index=False)
    joblib.dump({"clf": clf, "colunas": cols}, ARQ_MODELO)
    print("\nArquivos gerados: matriz_confusao.png, importancia_features.png, "
          f"resultados_teste.csv, {ARQ_MODELO}")


# ---------------- Previsão ao vivo ----------------
def baixar_json(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def enviar_modelo(chave, classe, prob_pct, estado_atual):
    q = urllib.parse.urlencode({"api_key": chave, "field1": classe,
                                "field2": f"{prob_pct:.1f}", "field3": estado_atual,
                                "status": "NoiseGuard RF"})
    for _ in range(3):
        with urllib.request.urlopen("https://api.thingspeak.com/update?" + q,
                                    timeout=20) as r:
            resp = r.read().decode().strip()
        if resp != "0":
            return resp
        time.sleep(16)          # limite do plano gratuito: 15 s entre envios
    return "0"


def ao_vivo(args):
    chave = os.environ.get("THINGSPEAK_ML_WRITE_KEY")
    if not chave:
        sys.exit("Defina a variável THINGSPEAK_ML_WRITE_KEY com a Write API Key "
                 "do canal do modelo (veja o cabeçalho deste arquivo).")
    if not args.canal:
        sys.exit("Informe --canal com o ID do canal do sensor.")
    m = joblib.load(ARQ_MODELO)
    clf, cols = m["clf"], m["colunas"]
    idx_acima = list(clf.classes_).index(2) if 2 in clf.classes_ else None
    renomear = {f"field{i + 1}": n for i, n in enumerate(CAMPOS)}
    print("Previsão ao vivo iniciada. Ctrl+C para parar.")

    while True:
        try:
            url = (f"https://api.thingspeak.com/channels/{args.canal}/feeds.json?results=60"
                   + (f"&api_key={args.read_key}" if args.read_key else ""))
            df = limpar(pd.DataFrame(baixar_json(url)["feeds"]).rename(columns=renomear),
                        verbose=False)
            idade = ((pd.Timestamp.now(tz="UTC") - df["created_at"].iloc[-1]).total_seconds()
                     if len(df) else None)
            if idade is None or idade > 120:
                print("ESP offline ou sem dados recentes: nada enviado.")
            else:
                f, _ = montar_features(df)
                ult = f.dropna().tail(1)
                if ult.empty or ult.index[-1] != df.index[-1]:
                    print("Ainda sem 5 min de histórico contínuo: aguardando.")
                else:
                    classe = int(clf.predict(ult[cols])[0])
                    prob = (clf.predict_proba(ult[cols])[0][idx_acima] * 100
                            if idx_acima is not None else 0.0)
                    atual = int(df["estado_baseline"].iloc[-1])
                    entrada = enviar_modelo(chave, classe, prob, atual)
                    print(f"{time.strftime('%H:%M:%S')} | agora: {NOMES_CLASSE[atual]} | "
                          f"em 2 min: {NOMES_CLASSE[classe]} | P(acima) = {prob:.0f}% | "
                          f"entrada {entrada}")
        except KeyboardInterrupt:
            print("\nEncerrado.")
            return
        except Exception as e:      # rede caiu, ThingSpeak lento etc.
            print("Erro:", e)
        time.sleep(args.intervalo)


# ---------------- Programa principal ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NoiseGuard - ML")
    ap.add_argument("modo", choices=["treinar", "aovivo"])
    ap.add_argument("--canal", help="Channel ID do canal do sensor")
    ap.add_argument("--read-key", help="Read API Key (só se o canal for privado)")
    ap.add_argument("--csv", help="CSV exportado do ThingSpeak")
    ap.add_argument("--intervalo", type=int, default=30,
                    help="segundos entre previsões ao vivo (mín. 20)")
    a = ap.parse_args()
    a.intervalo = max(a.intervalo, 20)
    treinar(a) if a.modo == "treinar" else ao_vivo(a)