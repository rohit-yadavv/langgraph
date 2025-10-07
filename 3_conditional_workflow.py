from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from pydantic import BaseModel, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" )


class ReviewState(TypedDict):
  review:str
  sentiment: Literal["positive", "negative"] 
  diagnosis: dict
  response: str


class SentimentSchema(BaseModel):
  sentiment: Literal["positive", "negative"]= Field(description="Sentiment of the review")

def find_sentiment(state: ReviewState):
  structured_model = model.with_structured_output(SentimentSchema)
  prompt = f"Analyze the sentiment of the following review: {state['review']}"
  response = structured_model.invoke(prompt)
  return {"sentiment": response.sentiment}


def check_sentiment(state: ReviewState):
  if state['sentiment'] == "positive":
    return "positive_response"
  else:
    return "diagnose_negative_issue"


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

def diagnose_negative_issue(state:ReviewState):
  structured_model = model.with_structured_output(DiagnosisSchema)
  prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
  "Return issue_type, tone, and urgency.
  """
  result = structured_model.invoke(prompt)

  return {
    "diagnosis":{
      "issue_type": result.issue_type,
      "tone": result.tone,
      "urgency": result.urgency
    }
  }

def positive_response(state: ReviewState):
  prompt = f"Generate a positive response for the following review: {state['review']}"
  response = model.invoke(prompt)
  return {"response": response.content}

def negative_response(state: ReviewState):
  diagnosis = state['diagnosis']

  prompt = f"""You are a support assistant.
  The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
  Write an empathetic, helpful resolution message.
  """
  response = model.invoke(prompt).content

  return {'response': response}


graph = StateGraph(ReviewState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("diagnose_negative_issue", diagnose_negative_issue)

graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response", END)

graph.add_edge("diagnose_negative_issue", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
result = workflow.invoke(intial_state)
print(result)