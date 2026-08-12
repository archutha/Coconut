from concurrent.futures import ThreadPoolExecutor

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

agent_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

TOURISM_PROMPT = """You are Coconut's tourism specialist. You only discuss summer travel
destinations and trips. Use the search tool for current, accurate information.
If the user asks for the answer in a specific language, respond in that language."""

FESTIVALS_PROMPT = """You are Coconut's festivals specialist. You only discuss summer festivals
and events. Use the search tool for current, accurate information.
If the user asks for the answer in a specific language, respond in that language."""

FASHION_PROMPT = """You are Coconut's fashion specialist. You only discuss summer fashion,
style, and clothing trends. Use the search tool for current, accurate information.
If the user asks for the answer in a specific language, respond in that language."""

tourism_agent = create_agent(agent_llm, tools=[search_summer_web], system_prompt=TOURISM_PROMPT)
festivals_agent = create_agent(agent_llm, tools=[search_summer_web], system_prompt=FESTIVALS_PROMPT)
fashion_agent = create_agent(agent_llm, tools=[search_summer_web], system_prompt=FASHION_PROMPT)

GREETING_MESSAGE = "Hey there! 🥥 I'm Coconut — ask me about summer tourism, festivals, or fashion!"
REFUSAL_MESSAGE = (
    "I can only help with summer tourism, festivals, and fashion! "
    "Ask me something in one of those areas."
)

_AGENTS = {
    "tourism": tourism_agent,
    "festivals": festivals_agent,
    "fashion": fashion_agent,
}

SYNTHESIS_PROMPT = """You are combining answers from multiple specialists to answer a user's
question about summer topics. Weave their information into ONE natural, coherent answer —
don't just list them separately or mention that there were multiple specialists.

Original question: {question}

{specialist_answers}
"""

def _invoke_agent(category: str, question: str) -> tuple[str, str]:
    agent = _AGENTS[category]
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return category, result["messages"][-1].content

def synthesize_answers(question: str, answers: dict) -> str:
    specialist_text = "\n\n".join(
        f"{category.capitalize()} specialist's info:\n{answer}"
        for category, answer in answers.items()
    )
    prompt = SYNTHESIS_PROMPT.format(question=question, specialist_answers=specialist_text)
    response = agent_llm.invoke(prompt)
    return response.content

def ask_coconut(question: str) -> str:
    categories = classify_question(question)

    if not categories:
        return REFUSAL_MESSAGE
    if categories == ["greeting"]:
        return GREETING_MESSAGE

    topic_categories = [c for c in categories if c != "greeting"]

    if len(topic_categories) == 1:
        _, answer = _invoke_agent(topic_categories[0], question)
        return answer

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_invoke_agent, c, question) for c in topic_categories]
        answers = dict(f.result() for f in futures)

    return synthesize_answers(question, answers)


if __name__ == "__main__":
    print(ask_coconut("Tell me what to wear for summer festivals happening in Spain"))