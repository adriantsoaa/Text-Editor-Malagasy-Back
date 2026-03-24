import joblib
import pandas as pd, re, pybktree
from pathlib import Path
from pydantic import BaseModel
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurez les origines autorisées (ne pas mettre de slash final)
origins = [
    "https://examen-machine-learning-m2-s1-front.vercel.app",
    #"http://localhost:3000", # Pour vos tests locaux
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

 # Remplacez ce chemin par le chemin réel de votre modèle IA exporté (ex: .joblib, .pkl, .pt)
# MODEL_PATH = Path("model.joblib")
# model = None

# def load_model():
#     global model
#     if MODEL_PATH.exists():
#         model = joblib.load(MODEL_PATH)
#     else:
#         print(f"[ERREUR] Le modèle IA n'a pas été trouvé à l'emplacement : {MODEL_PATH.resolve()}")
#         model = None

# load_model()

# # Exemple de schéma d'entrée pour la prédiction
# class InputData(BaseModel):
#     feature1: float
#     feature2: float

# # Endpoint racine
# @app.get("/")
# def read_root():
#     return {"message": "API Malagasy Text Editor is running!"}

# # Endpoint de prédiction exemple
# @app.post("/predict")
# def predict(data: InputData):
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded.")
#     # Adapter la forme des données selon votre modèle
#     X = [[data.feature1, data.feature2]]
#     prediction = model.predict(X)
#     return {"prediction": prediction[0]}

# ---------- Modèles Pydantic ----------
class PhraseRequest(BaseModel):
    phrase: str

class TokenResult(BaseModel):
    token: str
    statut: str        # "correct" | "erreur" | "ponctuation"
    suggestions: list[str]

class CorrectionResponse(BaseModel):
    resultats: list[TokenResult]

# ---------- Chargement dictionnaire ----------
def charger_dictionnaire() -> frozenset:
    df = pd.read_csv("L.csv")
    df.columns = ["racine", "derives"]
    mots = set()
    for _, row in df.iterrows():
        racine = re.sub(r"\s*\(.*?\)", "", str(row["racine"])).strip().lower()
        mots.add(racine)
        if pd.notna(row["derives"]):
            for d in str(row["derives"]).split(","):
                mots.add(d.strip().lower())
    return frozenset(mots)

STOPWORDS_MG = {"ny", "sy", "fa", "ka", "no", "na", "dia", "ary", "ho", "ao", "izay"}

def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

# Chargement au démarrage (une seule fois)
dictionnaire  = charger_dictionnaire()
vocabulaire   = dictionnaire | STOPWORDS_MG
bk            = pybktree.BKTree(levenshtein, dictionnaire)

# ---------- Logique de correction ----------
def corriger_phrase(phrase: str, max_dist: int = 2, top_n: int = 3) -> list[TokenResult]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+|[.,!?;:\-]", phrase)
    resultats = []
    for token in tokens:
        if re.match(r"^[.,!?;:\-]+$", token):
            resultats.append(TokenResult(token=token, statut="ponctuation", suggestions=[]))
            continue
        mot = token.lower()
        if mot in vocabulaire:
            resultats.append(TokenResult(token=token, statut="correct", suggestions=[]))
        else:
            candidats  = bk.find(mot, max_dist)
            suggestions = [w for _, w in sorted(candidats)][:top_n]
            resultats.append(TokenResult(token=token, statut="erreur", suggestions=suggestions))
    return resultats

# ---------- Route ----------
@app.post("/corriger", response_model=CorrectionResponse)
async def corriger(body: PhraseRequest):
    if not body.phrase.strip():
        raise HTTPException(status_code=400, detail="Phrase vide")
    return CorrectionResponse(resultats=corriger_phrase(body.phrase))