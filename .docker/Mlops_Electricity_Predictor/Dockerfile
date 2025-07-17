# Utilise une image de base légère
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet dans le conteneur
COPY . .

# Exposer le port utilisé par Uvicorn
EXPOSE 8000

# Commande pour démarrer l’API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
