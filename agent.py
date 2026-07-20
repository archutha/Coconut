from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from Router import classify_question

load_dotenv()

_search = DuckDuckGoSearchRun()

@tool
def search_summer_web(query: str) -> str:
    """Search the web for current information about summer tourism, festivals, or fashion.
    Use this whenever the question needs up-to-date or specific real-world information."""
    return _search.run(query)

COCONUT_SYSTEM_PROMPT = """You are Coconut, a friendly assistant that only discusses summer
tourism, summer festivals, and summer fashion. Use the search tool to find current,
accurate information rather than relying only on what you already know. If you're
unsure whether something fits your scope, err on the side of answering only within
tourism, festivals, or fashion."""

agent_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
coconut_agent = create_agent(
    agent_llm,
    tools=[search_summer_web],
    system_prompt=COCONUT_SYSTEM_PROMPT,
)

REFUSAL_MESSAGE = (
    "I can only help with summer tourism, festivals, and fashion! "
    "Ask me something in one of those areas."
)

GREETING_MESSAGE = "Hey there! 🥥 I'm Coconut — ask me about summer tourism, festivals, or fashion!"


def ask_coconut(question: str) -> str:
    category = classify_question(question)
    if category == "greeting":
        return GREETING_MESSAGE
    if category == "off_topic":
        return REFUSAL_MESSAGE

    result = coconut_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask_coconut("What festivals happen in Spain during summer?"))
    print(ask_coconut("What's the capital of France?"))