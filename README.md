# 📊 DTA_BOT – AI Data Analyst

An **AI-powered Data Analyst Assistant** that automatically analyzes datasets, generates visualizations, extracts insights using LLMs, and trains machine learning models.

Built using **Python, Streamlit, LangChain, Groq LLMs, and Scikit-Learn**.

---

# 🚀 Features

## 📂 Upload Dataset
Upload any CSV file and instantly explore your dataset.

## 📊 Automatic Data Visualization
The system automatically generates:

- Line Charts
- Bar Charts

Graphs are generated using **Matplotlib**.

---

## 🤖 AI Chart Analysis
Charts are analyzed using a **Large Language Model** to generate:

- Insights
- Trends
- Conclusions

Powered by **Groq + LangChain structured outputs**.

---

## 🧠 Machine Learning Training

Train models automatically with **Random Forest + Hyperparameter Search**.

Supports:

- Regression
- Classification

Includes:

- Data preprocessing
- Missing value imputation
- Feature scaling
- One-hot encoding
- RandomizedSearchCV

Implemented using **Scikit-Learn pipelines**.

---

# 🏗 Project Structure
DTA_BOT
│
├── data_analyst_bot.py # Main Streamlit application
├── Graph.py # Graph generation
├── Analysis.py # LLM-based chart analysis
├── ML_algo.py # Machine learning pipeline
│
├── Graphs/
│ ├── line_graph.png
│ └── bar_graph.png
│
├── model.pkl # Saved trained model
├── requirements.txt
└── README.md

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/DTA_BOT.git
cd DTA_BOT
2️⃣ Create Virtual Environment
python -m venv venv

Activate environment

Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add Environment Variables

Create a .env file

GROQ_API_KEY=your_api_key
▶️ Run Application
streamlit run data_analyst_bot.py

Then open in browser

http://localhost:8501
📈 Workflow

Upload CSV dataset

Select target column

Generate visualizations

AI analyzes charts

Train ML model automatically

Download trained model

🛠 Tech Stack

Python

Streamlit

LangChain

Groq LLM

Scikit-Learn

Matplotlib

Pandas

Pydantic

🎯 Future Improvements

Auto feature engineering

Auto model selection (AutoML)

Interactive dashboards

Forecasting models

Export insights report (PDF)

👨‍💻 Author

Anurag

AI Engineer | Machine Learning Developer

GitHub: https://github.com/Anurag07-crypto

⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
🚀 Build something amazing