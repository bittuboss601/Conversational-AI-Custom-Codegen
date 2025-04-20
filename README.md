# 🤖 Conversational AI | Custom CodeGen Chatbot

A powerful **LLM-powered Code Generator and Debugger chatbot** built using the Rasa Framework, FastAPI, and a fine-tuned version of the [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono) model.

### 🔥 Hosted model on Hugging Face:
👉 [tiwaripurnendu121/codegen-350M-mbpp-csn](https://huggingface.co/tiwaripurnendu121/codegen-350M-mbpp-csn)

---

## 🚀 Features

- ✨ **AI-Powered Code Generation** (Python, Java, SQL, JS)
- 🐞 **Code Debugging + Explanation**
- ⚙️ **Rasa-powered conversational interface**
- ⚡ **FastAPI backend to serve LLM responses**
- 📚 Trained on **MBPP + CodeSearchNet datasets**
- 🔌 Easily pluggable into Web, WhatsApp, Facebook, etc.

---

## 🧠 Tech Stack

- `Rasa` for NLU, dialogue management & custom actions
- `FastAPI` for hosting CodeGen inference server
- `Transformers` for loading and generating from Hugging Face model
- `Hugging Face Hub` for fine-tuned LLM storage
- `Python`, `Docker`, `Colab` for training pipeline

---

## 📦 Folder Structure

Conversational-AI-Custom-Codegen/ ├── actions/ # Custom Rasa actions calling CodeGen API ├── data/ # Rasa NLU training data ├── models/ # Rasa models ├── notebooks/ # Colab fine-tuning + deployment notebook ├── server/ # FastAPI server to serve model responses ├── domain.yml # Rasa domain config ├── config.yml # Rasa pipeline config ├── endpoints.yml # Rasa endpoints (for custom actions) ├── credentials.yml # Channel connectors ├── docker-compose.yml # Optional: For containerization └── README.md 


---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/bittuboss601/Conversational-AI-Custom-Codegen.git
cd Conversational-AI-Custom-Codegen

cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

rasa train
rasa run actions & rasa shell

rasa run --enable-api --cors "*"

Type | Example Prompt
Generate | Write code for a REST API in Flask
Debug | Debug and explain the following Python code:
Multi-lang | Generate a SQL query to fetch user details
```

### 📬 Connect With Me

Let’s connect and collaborate on AI-powered tools!

🔗 LinkedIn: [Purnendu Tiwari](https://www.linkedin.com/in/purnendu-tiwari/)

🤗 Hugging Face: [tiwaripurnendu121](https://huggingface.co/tiwaripurnendu121/codegen-350M-mbpp-csn)
