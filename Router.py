from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class TopicClassification(BaseModel):
    category: Literal["tourism", "festivals", "fashion", "greeting", "off_topic"] = Field(
        description="Which single category the user's question belongs to."
    )

router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
router = router_llm.with_structured_output(TopicClassification)

def classify_question(question: str) -> str:
    result = router.invoke(
        f"""Classify this question into exactly one category: tourism, festivals, fashion, greeting, or off_topic.
Use "greeting" for simple hellos, thanks, goodbyes, or casual pleasantries with no real topic.
Only use tourism, festivals, or fashion if the question is specifically about SUMMER activities —
not winter, not general/unrelated topics. Anything else genuinely unrelated is off_topic.


Question: {question}"""
    )
    return result.category


if __name__ == "__main__":
    print(classify_question("What festivals happen in Spain during summer?"))
    print(classify_question("What's the capital of France?"))
    print(classify_question("Best sunscreen for oily skin in summer?"))