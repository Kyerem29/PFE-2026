#  Assistant RAG — Data Science (PFE 2026)

Chatbot intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)** permettant d'interroger des documents PDF liés à la Data Science (Matplotlib, Seaborn, Pandas…).

Développé avec **Gradio**, **LangChain**, **ChromaDB** et **Ollama (Mistral)**.

---

## ✨ Fonctionnalités

-  Interface de chat multi-sessions avec historique personnel par utilisateur
-  Réponses basées sur vos propres documents PDF (RAG)
-  Authentification par utilisateur/mot de passe
-  Persistance des conversations en base SQLite
-  Affichage des sources PDF utilisées pour chaque réponse
-  Interface stylisée (thème custom + police DM Sans)

---

##  Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Gradio UI  │────▶│  LangChain   │────▶│   Mistral   │
│  (Frontend) │     │  RAG Chain   │     │  (via Ollama)│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   ChromaDB   │
                    │  (PDF index) │
                    └─────────────┘
```

---

##  Installation

### 1. Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/download) installé sur la machine

### 2. Télécharger les modèles Ollama

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 3. Cloner le repo et installer les dépendances

```bash
git clone https://github.com/Kyerem29/PFE-2026.git
cd PFE-2026
pip install -r requirements.txt
```

### 4. Ajouter vos documents PDF

Placez vos fichiers `.pdf` dans le dossier `data/` :

```
PFE-2026/
├── data/
│   ├── matplotlib_doc.pdf
│   ├── seaborn_tutorial.pdf
│   └── ...
```

### 5. Lancer l'application

```bash
python "Chatbot PFE.py"
```

L'interface sera accessible sur `http://localhost:7860`

---

##  Identifiants de connexion

| Utilisateur | Mot de passe |
|-------------|--------------|
| kevin       | pfe2026      |
| admin       | pfe2026      |

> ⚠️ Pensez à changer ces identifiants avant tout déploiement en production.

---

##  Stack technique

| Composant | Technologie |
|-----------|-------------|
| Interface | Gradio |
| LLM | Mistral 7B via Ollama |
| Embeddings | nomic-embed-text |
| RAG | LangChain |
| Vector store | ChromaDB |
| Base de données | SQLite |

---

##  Structure du projet

```
PFE-2026/
├── Chatbot PFE.py        # Application principale
├── requirements.txt      # Dépendances Python
├── data/                 # Dossier des documents PDF
├── chroma_db_matplotlib/ # Index vectoriel (généré automatiquement)
└── chat_history.db       # Historique des conversations (généré automatiquement)
```

---

##  Auteur

**Kyerem29** — PFE 2026
