import joblib
from pathlib import Path
from pydantic import BaseModel
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurez les origines autorisées (ne pas mettre de slash final)
origins = [
    "https://examen-machine-learning-m2-s1-front.vercel.app",
    "http://localhost:3000", # Pour vos tests locaux
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 # Remplacez ce chemin par le chemin réel de votre modèle IA exporté (ex: .joblib, .pkl, .pt)
MODEL_PATH = Path("model.joblib")
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        print(f"[ERREUR] Le modèle IA n'a pas été trouvé à l'emplacement : {MODEL_PATH.resolve()}")
        model = None

load_model()

# Exemple de schéma d'entrée pour la prédiction
class InputData(BaseModel):
    feature1: float
    feature2: float

# Endpoint racine
@app.get("/")
def read_root():
    return {"message": "API Malagasy Text Editor is running!"}

# Endpoint de prédiction exemple
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    # Adapter la forme des données selon votre modèle
    X = [[data.feature1, data.feature2]]
    prediction = model.predict(X)
    return {"prediction": prediction[0]}