# --------------------------------------------------------
import pandas as pd 
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from Graph import line_graph, bar_graph
from ML_algo import rf_param, model_train
from Analysis import analysis
import streamlit as st 
import os  
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# ////////////////////////////////////////////////////////
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful data analyst."),
    ("human","Analyze this chart: {Message}")
])

load_dotenv()
os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2)

chain = prompt | llm

class structure(BaseModel):
    Insights:str = Field(description="Give Insights of graphs")
    Trends:str = Field(description="Find Trends and provide me trends")
    Conclusion:str = Field(description="Provide conclustion of it")

# ////////////////////////////////////////////////////////
# --------------------------------------------------------

rf_reg, rf_class = rf_param()
# --------------------------------------------------------
st.title("📈 DTA_BOT")
st.subheader("🤖 Your Virtual Data Analyst")
st.write("Let's start")
st.text("Drop your working csv")
uploaded_file = st.file_uploader("Upload here", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("View your data")
    with st.expander("See your data"):
        st.dataframe(df)
    columns =df.columns.to_list()
    X = df.drop(columns=[columns[-1]])
    y_choice = st.selectbox("Select target column",[None] + columns)
    if y_choice is not None:
        y = df[y_choice]
# /////////////////////////////////////////////////////////////////////////////////
        x_axis = st.selectbox("select x_axis",df.columns.to_list())
        col_1, col_2 = st.columns(2)
        if x_axis and st.button("Start analysis"):
            with col_1: # line graph
                    line_graph(df=df, x_axis=x_axis, y_axis=y_choice)
                    st.image("DTA_BOT/Graphs/line_graph.png")
            with col_2: # bar graph
                bar_graph(df=df, x_axis=x_axis, y_axis=y_choice)
                st.image("DTA_BOT/Graphs/bar_graph.png")
            st.title("Graph Analysis")
            insights, trends, conclusion = analysis(X_axis=df[x_axis], y_axis=df[y_choice])
            # chart_2 = analysis("DTA_BOT/Graphs/bar_graph.png")
            message = f"""
Insights: {insights}

Trends: {trends}

Conclusion: {conclusion}
"""
            response = chain.invoke({"Message":message})
            st.markdown(f'''Here are the Analysis\n
                        {response.content}
                        ''')
# /////////////////////////////////////////////////////////////////////////////////
    choice = st.radio("Choose your data label",["Regression", "Classification"])
    if choice=="Regression":
        if st.button("Click to train model"): 
            model, score, message = model_train(rf_param_grid=rf_reg, X=X, y=y,choice=choice)
            st.header(message)
            st.write("Model score is ")
            st.text(score)
            if model:
                st.write("Now use the model in your system")
                st.write("Here is path model.pkl")
                
    if choice=="Classification":
        if st.button("Click to train model"):
            model, score, message = model_train(rf_param_grid=rf_class, X=X, y=y,choice=choice)
            st.header(message)
            st.write("Model score is ")
            st.text(score*100)
            if model:
                st.write("Now use the model in your system")
                st.write("Here is path model.pkl")
                
            