from crewai import LLM, Agent 
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


## call the gemini models
llm=LLM(
    model="gemini/gemini-2.0-flash-thinking-exp-1219",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GEMINI_API_KEY"),
    
)

## creating a write agent with custom
programer = Agent(
  role='sinior programer',
  goal='Create a web page based on a description.',
  verbose=True,
  memory=True,
  backstory=(
    "Skilled programmer who does professional programming on demand"
    "Very accurate, writes the code through the description with all professionalism"
  ),
  llm=llm,
  allow_delegation=False
)
