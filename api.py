import joblib
import csv, os
import pandas as pd, re, pybktree
from pathlib import Path
from pydantic import BaseModel
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(redirect_slashes=False)

# Configurez les origines autorisées (ne pas mettre de slash final)
origins = [
    "https://examen-machine-learning-m2-s1-front.vercel.app",
    #"http://localhost:3000", # Pour vos tests locaux
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    df = pd.read_csv("dataset/DatasetMalagasy.csv", encoding="latin-1")
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

# ---------- Modèles Sentiment ----------
class SentimentResponse(BaseModel):
    label: str        # "Positif" | "Négatif" | "Neutre"
    score: int
    positifs: int
    negatifs: int

# ---------- Dictionnaires ----------
positive_words = {"tsara", "soa", "mahafinaritra", "tia", "faly", "sambatra", "mahafaly", "fitiavana", "firindrana", "fanantenana", "maniry", "mankasitraka", "midera", "manaja", "miara", "mahery", "matanjaka", "manan-kery", "hendry", "mahalala", "malala", "mazava", "marina", "mendrika", "vonona", "velona", "mifaly", "miorina", "ampy", "nahomby", "voaaro", "afaka", "mahay", "maivana", "manitra", "tonga", "mirindra", "mahatoky", "zara", "mifankatia", "miray", "mahasoa", "mahomby", "mahitsy", "mamelona", "mampandroso", "fandroso", "fandrosoana", "rariny", "fahamarinana", "fahendrena", "fahaizana", "fananana", "hafaliana", "hasambarana", "fifankatiavana", "fitiavam-pirenena", "fahatokiana", "fiadanana", "fiarahana", "firaisana", "fivavahana", "fahasoavana", "fitahiana", "vonjy", "tohana", "mazoto", "tsiky", "herimpo", "sahy", "malina"}

negative_words = {"ratsy", "marary", "malahelo", "kivy", "tahotra", "alahelo", "tezitra", "sahirana", "maizina", "loza", "diso", "reraka", "malazo", "mahantra", "fadiranovana", "haromotana", "fahoriana", "fanafihana", "miady", "mamono", "mandroba", "mandainga", "mampahory", "mahamenatra", "very", "mampanahy", "osa", "kamo", "adala", "adiana", "masiaka", "mahafaty", "manimba", "mifanditra", "mifanandrina", "manary", "taitra", "mitaraina", "tsiny", "henatra", "latsa", "faneriterena", "fanenjehana", "fahantrana", "fahalemena", "fahasarotana", "fahaketrahana", "faharesena", "voatery", "voaroba", "maloaka"}

def analyser_sentiment(phrase: str) -> SentimentResponse:
    tokens = re.sub(r"[.,!?;:'\"()\-]", " ", phrase.lower()).split()
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    score = pos - neg
    label = "Positif" if score > 0 else ("Négatif" if score < 0 else "Neutre")
    return SentimentResponse(label=label, score=score, positifs=pos, negatifs=neg)

# ---------- Route Sentiment ----------
@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(body: PhraseRequest):
    if not body.phrase.strip():
        raise HTTPException(status_code=400, detail="Phrase vide")
    return analyser_sentiment(body.phrase)


# ---------- Lemmatiseur ----------
class MalagasyUltraAnalyzer:
    def __init__(self, file_path='dataset/DatasetMalagasy.csv'):
        self.lemma_map = {}
        self.all_roots = set()
        self.stop_words = {
            'ny','sy','dia','ary','fa','nefa','kanefa','hoe','izay','no','ka',
            'tamin','an','eo','ao','any','ho','mba','ireo','ity','izany','io',
            'izao','tokony','efa','mbola','koa','rehetra','izy','izahay','ianareo','isika'
        }
        self.prefixes = sorted(['mpan','mpam','maha','man','mam','mi','ma','fi','fan','fam','mpi'], key=len, reverse=True)
        self.suffixes = sorted(['ana','ina','na'], key=len, reverse=True)
        self.infixes  = ['in','om','im','if','re','il','amp']
        self.load_data(file_path)

    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, mode='r', encoding='latin1') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        root = row[0].strip().lower()
                        self.all_roots.add(root)
                        for d in str(row[1]).split(','):
                            d = d.strip().lower()
                            if d: self.lemma_map[d] = root
                        self.lemma_map[root] = root

    def analyze_word(self, word: str):
        original = word.lower().strip().strip('.,!?:;()[]"\'')
        if not original or original in self.stop_words:
            return None
        prefix, suffix, infix, current = "", "", "", original
        for p in self.prefixes:
            if current.startswith(p):
                prefix, current = p, current[len(p):]
                break
        for s in self.suffixes:
            if current.endswith(s):
                suffix, current = s, current[:-len(s)]
                break
        final_root, found_infix = current, ""
        for i in self.infixes:
            if i in current[1:4]:
                parts = current.split(i, 1)
                reconstructed = parts[0] + parts[1]
                if reconstructed in self.all_roots or reconstructed in self.lemma_map:
                    found_infix, final_root = i, reconstructed
                    break
        if final_root == current and original in self.lemma_map:
            final_root = self.lemma_map[original]
        return {"mot": original, "prefixe": prefix, "infixe": found_infix, "racine": final_root, "suffixe": suffix}

lemmatiseur = MalagasyUltraAnalyzer('dataset/DatasetMalagasy.csv')

# ---------- Modèles Pydantic ----------
class WordAnalysis(BaseModel):
    mot: str
    prefixe: str
    infixe: str
    racine: str
    suffixe: str
    est_stop_word: bool

class LemmaResponse(BaseModel):
    resultats: list[WordAnalysis]

# ---------- Route ----------
@app.post("/lemmatiser", response_model=LemmaResponse)
async def lemmatiser(body: PhraseRequest):
    if not body.phrase.strip():
        raise HTTPException(status_code=400, detail="Phrase vide")
    import re
    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", body.phrase)
    resultats = []
    for token in tokens:
        res = lemmatiseur.analyze_word(token)
        if res is None:
            resultats.append(WordAnalysis(mot=token.lower(), prefixe="", infixe="", racine=token.lower(), suffixe="", est_stop_word=True))
        else:
            resultats.append(WordAnalysis(**res, est_stop_word=False))
    return LemmaResponse(resultats=resultats)
