"""Chat routes backed by LangGraph."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.graphs.daily import GraphKind, get_graph
from agents.limiter import limiter
from agents.logging_setup import get_logger, log_payload
from agents.settings import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger("agents.chat")


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8000)
    graph: GraphKind = "simple"


class ChatResponse(BaseModel):
    thread_id: str
    graph: str
    response: str


def _serialize_message(message) -> dict:
    return {
        "type": message.__class__.__name__,
        "content": message.content or "",
        "name": getattr(message, "name", None),
    }


@router.post("", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured",
        )

    graph = get_graph(body.graph, settings)
    config = {"configurable": {"thread_id": body.thread_id}}
    log_payload(
        logger,
        "chat.request",
        body.model_dump(),
    )

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        log_payload(
            logger,
            "chat.error",
            {"thread_id": body.thread_id, "graph": body.graph, "error": error},
            level=logging.ERROR,
        )
        raise HTTPException(status_code=500, detail=error) from exc

    last = result["messages"][-1]
    response = ChatResponse(
        thread_id=body.thread_id,
        graph=body.graph,
        response=last.content or "",
    )
    log_payload(logger, "chat.response", response.model_dump())
    return response


@router.get("/threads/{thread_id}")
@limiter.limit("60/minute")
async def get_thread(request: Request, thread_id: str, graph: GraphKind = "simple"):
    settings = get_settings()
    compiled = get_graph(graph, settings)
    config = {"configurable": {"thread_id": thread_id}}
    state = await compiled.aget_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = state.values.get("messages", [])
    return {
        "thread_id": thread_id,
        "graph": graph,
        "message_count": len(messages),
        "messages": [_serialize_message(m) for m in messages],
    }
