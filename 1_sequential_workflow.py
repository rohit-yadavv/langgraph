from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" )


class LLMState(TypedDict):
  question:str
  answer:str

def ask_question(state:LLMState):
  question = input("Ask your Question: ")
  
  # state["question"] = question
  # return state
  return {"question":question}

def answer_question(state:LLMState):
  answer = model.invoke(state["question"])
  return {"answer":answer}


graph_builder = StateGraph(LLMState)
graph_builder.add_node("question",ask_question)
graph_builder.add_node("answer",answer_question)

graph_builder.add_edge(START, "question")
graph_builder.add_edge("question", "answer")
graph_builder.add_edge("answer", END)


graph  = graph_builder.compile()

initial_state = {}
result = graph.invoke(initial_state)
print(result)