from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class TopicClassification(BaseModel):
    categories: list[Literal["tourism", "festivals", "fashion", "greeting"]] = Field(
        description="""All topics this question relates to. Include more than one if the
        question genuinely spans multiple areas (e.g., "what to wear to a summer festival"
        = both fashion and festivals). Use only "greeting" for simple hellos/thanks with no
        real topic. Return an empty list if the question is unrelated to summer tourism,
        festivals, or fashion."""
    )

router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
router = router_llm.with_structured_output(TopicClassification)

def classify_question(question: str) -> list[str]:
    result = router.invoke(f"Classify this question: {question}")
    return result.categories