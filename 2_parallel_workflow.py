import operator
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash" )


class UPSCState(TypedDict):
  topic:str
  essay:str
  language_feedback: str
  analysis_feedback: str
  clarity_feedback: str
  overall_feedback: str
  individual_scores: Annotated[list[int], operator.add]
  avg_score: float


class EvaluationScehma(BaseModel):
  feedback: str
  score: int

structured_model = model.with_structured_output(EvaluationScehma)


def generate_essay(state:UPSCState):
  res = model.invoke(f"Generate an essay on {state['topic']}")
  return {"essay":res.content}

def generate_language_feedback(state: UPSCState):
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def generate_analysis_feedback(state: UPSCState):
  prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
  result = structured_model.invoke(prompt)

  return {'analysis_feedback': result.feedback, 'individual_scores': [result.score]}

def generate_clarity_feedback(state: UPSCState):
  prompt = f'Evaluate the clarity of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
  result = structured_model.invoke(prompt)
  return {'clarity_feedback': result.feedback, 'individual_scores': [result.score]}

def final_evaluation(state: UPSCState): 
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'

    overall_feedback = model.invoke(prompt).content
 
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}
    


graph = StateGraph(UPSCState)
graph.add_node("generate_essay", generate_essay)
graph.add_node("generate_language_feedback", generate_language_feedback)
graph.add_node("generate_analysis_feedback", generate_analysis_feedback)
graph.add_node("generate_clarity_feedback", generate_clarity_feedback)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "generate_essay")

graph.add_edge("generate_essay", "generate_language_feedback")
graph.add_edge("generate_essay", "generate_analysis_feedback")
graph.add_edge("generate_essay", "generate_clarity_feedback")

graph.add_edge("generate_language_feedback", "final_evaluation")
graph.add_edge("generate_analysis_feedback", "final_evaluation")
graph.add_edge("generate_clarity_feedback", "final_evaluation")

graph.add_edge("final_evaluation", END)

graph = graph.compile()

print(graph)

initial_state = {
  "topic": "The role of technology in modern education"
}

result = graph.invoke(initial_state)
print(result)