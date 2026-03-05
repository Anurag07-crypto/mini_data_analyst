from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
# ------------------------------------
from langchain_groq import ChatGroq


class structure(BaseModel):
    Insights:str = Field(description="Give Insights of graphs")
    Trends:str = Field(description="Find Trends and provide me trends")
    Conclusion:str = Field(description="Provide conclustion of it")

load_dotenv()
os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b",
        temperature=0.0,
        max_retries=2)
# converting images into base64
def analysis(X_axis, y_axis):
    
    prompt = f'''
    Analyze this chart and provide
    1: Insights
    2: Trends
    3: Conclusion
    here is needed X_axis {X_axis} and y_axis {y_axis}
    '''

    str_output = llm.with_structured_output(structure)
    response = str_output.invoke(prompt)
    insights = response.Insights
    trends = response.Trends
    conclusion = response.Conclusion
    return insights, trends, conclusion
