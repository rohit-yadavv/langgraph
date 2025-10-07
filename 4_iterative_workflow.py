import operator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel  

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro" )

# state
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


def generate_tweet(state: TweetState): 
  messages = [
    SystemMessage(content="You are a funny and clever Twitter/X influencer."),
    HumanMessage(content=f"""
      Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

      Rules:
      - Do NOT use question-answer format.
      - Max 280 characters.
      - Use observational humor, irony, sarcasm, or cultural references.
      - Think in meme logic, punchlines, or relatable takes.
      - Use simple, day to day english
    """)]
  res = model.invoke(messages)
  return {"tweet": res.content}

class EvaluationSchema(BaseModel):
  evaluation: Literal["approved", "needs_improvement"]
  feedback: str

def evaluate_tweet(state: TweetState):
  messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
    Evaluate the following tweet:

    Tweet: "{state['tweet']}"

    Use the criteria below to evaluate the tweet:

    1. Originality - Is this fresh, or have you seen it a hundred times before?  
    2. Humor - Did it genuinely make you smile, laugh, or chuckle?  
    3. Punchiness - Is it short, sharp, and scroll-stopping?  
    4. Virality Potential - Would people retweet or share it?  
    5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

    Auto-reject if:
    - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
    - It exceeds 280 characters
    - It reads like a traditional setup-punchline joke
    - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

    ### Respond ONLY in structured format:
    - evaluation: "approved" or "needs_improvement"  
    - feedback: One paragraph explaining the strengths and weaknesses 
  """)
  ]
  res = model.with_structured_output(EvaluationSchema).invoke(messages)
  return {"evaluation": res.evaluation, "feedback": res.feedback, "feedback_history": [res.feedback]}

def optimize_tweet(state: TweetState):

  messages = [
    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
    HumanMessage(content=f"""
      Improve the tweet based on this feedback:
      "{state['feedback']}"

      Topic: "{state['topic']}"
      Original Tweet:
      {state['tweet']}

      Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
      """)
  ]

  response = model.invoke(messages).content
  iteration = state['iteration'] + 1

  return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}


def route_evaluation(state: TweetState):
  if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
      return 'approved'
  else:
      return 'needs_improvement'

graph = StateGraph(TweetState)
graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("optimize_tweet", optimize_tweet)

graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")

graph.add_conditional_edges("evaluate_tweet", route_evaluation, {'approved':END, 'needs_improvement':'optimize_tweet'})

graph.add_edge("optimize_tweet", END)

graph = graph.compile()

initial_state = {
  'topic': 'The future of AI',
  'iteration': 1,
  'max_iteration': 3
}
result = graph.invoke(initial_state)
print(result)