from typing import List
from langchain.schema import Document

REFUSAL_OOS = "This question is outside the scope of my knowledge base."
REFUSAL_EMPTY = "I’m sorry, I don’t have that information in my knowledge base."

def outside_scope(docs: List[Document]) -> bool:
    return len(docs) == 0

def grounded_enough(answer: str, docs: List[Document]) -> bool:
    if not answer or len(answer.strip()) < 5:
        return False
    if not docs:
        return False
    # Require inline citations for groundedness (optional)
    if "(source:" not in answer:
        return False
    return True
