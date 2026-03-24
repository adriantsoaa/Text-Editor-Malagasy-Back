# Utiliser une image Python légère
FROM python:3.9

# Créer un utilisateur pour éviter les problèmes de permissions
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Définir le dossier de travail
WORKDIR /app

# Copier et installer les dépendances
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copier le reste du code
COPY --chown=user:user . .

# Lancer l'application sur le port 7860 (obligatoire sur HF)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]