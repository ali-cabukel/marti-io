"""LangGraph graphs for the daily agent hub."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from agents.checkpointer import get_checkpointer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langchain_core.tools import tool

from agents.integrations.dictionary import fetch_word_of_the_day
from agents.integrations.github import fetch_github_trending
from agents.integrations.news import fetch_news_headlines
from agents.integrations.weather import fetch_weather, format_weather
from agents.settings import Settings, get_settings


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return format_weather(fetch_weather(city))


@tool
def search_news(topic: str) -> str:
    """Search recent news for a topic."""
    settings = get_settings()
    return fetch_news_headlines(topic, api_key=settings.news_api_key)


@tool
def word_of_the_day() -> str:
    """Recommend an English word with definition and example usage."""
    return fetch_word_of_the_day()


@tool
def query_sqlite(sql: str) -> str:
    """Run a read-only SQLite query (demo)."""
    if not sql.strip().lower().startswith("select"):
        return "Only SELECT queries are allowed in this demo."
    return "Query accepted — connect a database path in a future release."


@tool
def github_trending(language: str = "") -> str:
    """List trending GitHub repositories, optionally filtered by language."""
    return fetch_github_trending(language)


TOOLS = [get_weather, search_news, word_of_the_day, query_sqlite, github_trending]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

DOMAIN_PROMPTS = {
    "weather": "You help with weather. Use get_weather when needed.",
    "news": "You summarize news. Use search_news when needed.",
    "dictionary": "You teach vocabulary. Use word_of_the_day when helpful.",
    "github": "You discuss repos. Use github_trending when needed.",
    "general": "You are a helpful daily assistant. Use tools when useful.",
}


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    domain: str


def _llm(settings: Settings):
    return _chat_model(settings).bind_tools(TOOLS)


def _chat_model(settings: Settings) -> ChatOpenAI:
    kwargs: dict[str, object] = {
        "model": settings.openai_model,
        "temperature": 0,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def _tool_node(state: ChatState):
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": []}
    results = []
    for call in last.tool_calls:
        tool_impl = TOOLS_BY_NAME[call["name"]]
        content = tool_impl.invoke(call["args"])
        results.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
    return {"messages": results}


def _should_tools(state: ChatState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _make_checkpointer(settings: Settings):
    try:
        return get_checkpointer()
    except RuntimeError:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def build_simple_graph(settings: Settings):
    """Single-agent ReAct loop — default for chat API."""
    graph = StateGraph(ChatState)

    def call_llm(state: ChatState):
        response = _llm(settings).invoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("llm", call_llm)
    graph.add_node("tools", _tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", _should_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile(checkpointer=_make_checkpointer(settings))


def _domains_for(text: str) -> list[str]:
    lowered = text.lower()
    domains: list[str] = []
    if any(k in lowered for k in ("weather", "temperature", "forecast")):
        domains.append("weather")
    if any(k in lowered for k in ("news", "headline", "market")):
        domains.append("news")
    if any(k in lowered for k in ("word", "vocabulary", "english")):
        domains.append("dictionary")
    if any(k in lowered for k in ("github", "repo", "repository")):
        domains.append("github")
    return domains or ["general"]


def build_map_reduce_graph(settings: Settings):
    """Fan out to domain workers in parallel, then synthesize one reply."""

    def router(state: ChatState):
        text = state["messages"][-1].content
        return [
            Send("worker", {"messages": state["messages"], "domain": domain})
            for domain in _domains_for(text)
        ]

    def worker(state: ChatState):
        domain = state["domain"]
        prompt = DOMAIN_PROMPTS[domain]
        response = _llm(settings).invoke(
            [SystemMessage(content=prompt), *state["messages"]]
        )
        if isinstance(response, AIMessage) and response.tool_calls:
            tool_msgs = _tool_node({"messages": [response]})["messages"]
            follow_up = _llm(settings).invoke([SystemMessage(content=prompt), *state["messages"], response, *tool_msgs])
            content = follow_up.content if isinstance(follow_up, AIMessage) else str(follow_up)
        else:
            content = response.content if isinstance(response, AIMessage) else str(response)
        return {"messages": [AIMessage(content=f"[{domain}] {content}", name=domain)]}

    def synthesize(state: ChatState):
        worker_msgs = [
            m for m in state["messages"] if isinstance(m, AIMessage) and getattr(m, "name", None)
        ]
        if not worker_msgs:
            answer = _chat_model(settings).invoke(state["messages"])
            return {"messages": [AIMessage(content=answer.content)]}

        combined = "\n".join(m.content or "" for m in worker_msgs)
        answer = _chat_model(settings).invoke(
            f"Synthesize these specialist notes into one concise reply:\n{combined}"
        )
        return {"messages": [AIMessage(content=answer.content)]}

    graph = StateGraph(ChatState)
    graph.add_node("worker", worker)
    graph.add_node("synthesize", synthesize)
    graph.set_conditional_entry_point(router)
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=_make_checkpointer(settings))


GraphKind = Literal["simple", "map_reduce"]

GRAPHS: dict[GraphKind, object] = {}


def get_graph(kind: GraphKind, settings: Settings):
    if kind not in GRAPHS:
        builder = build_simple_graph if kind == "simple" else build_map_reduce_graph
        GRAPHS[kind] = builder(settings)
    return GRAPHS[kind]
