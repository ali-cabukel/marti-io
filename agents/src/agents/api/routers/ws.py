"""WebSocket chat routes with Redis pub/sub and LangGraph streaming."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage

from agents.graphs.daily import GraphKind, get_graph
from agents.logging_setup import get_logger, log_payload, set_request_id, ws_log_level
from agents.settings import get_settings
from agents.ws.manager import get_ws_manager

router = APIRouter(prefix="/chat", tags=["chat-ws"])
logger = get_logger("agents.ws")


async def _stream_chat(
    *,
    thread_id: str,
    message: str,
    graph_kind: GraphKind,
) -> None:
    settings = get_settings()
    manager = get_ws_manager()
    graph = get_graph(graph_kind, settings)
    config = {"configurable": {"thread_id": thread_id}}

    await manager.publish(
        thread_id,
        {"type": "start", "thread_id": thread_id, "graph": graph_kind},
    )
    log_payload(
        logger,
        "ws.chat.start",
        {"thread_id": thread_id, "graph": graph_kind, "message": message},
    )

    full_response: list[str] = []

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            if event_type == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", None)
                if not content:
                    continue
                if isinstance(content, list):
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                else:
                    text = str(content)
                if not text:
                    continue
                full_response.append(text)
                await manager.publish(
                    thread_id,
                    {"type": "token", "content": text},
                )
            elif event_type == "on_tool_start":
                await manager.publish(
                    thread_id,
                    {
                        "type": "tool_start",
                        "name": event.get("name", ""),
                    },
                )
            elif event_type == "on_tool_end":
                output = event["data"].get("output", "")
                await manager.publish(
                    thread_id,
                    {
                        "type": "tool_end",
                        "name": event.get("name", ""),
                        "content": str(output),
                    },
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        log_payload(
            logger,
            "ws.chat.error",
            {"thread_id": thread_id, "graph": graph_kind, "error": error},
            level=logging.ERROR,
        )
        await manager.publish(
            thread_id,
            {"type": "error", "message": error},
        )
        return

    response_text = "".join(full_response)
    if not response_text:
        state = await graph.aget_state(config)
        messages = state.values.get("messages", []) if state.values else []
        if messages and isinstance(messages[-1], AIMessage):
            response_text = messages[-1].content or ""

    await manager.publish(
        thread_id,
        {
            "type": "done",
            "thread_id": thread_id,
            "graph": graph_kind,
            "response": response_text,
        },
    )
    log_payload(
        logger,
        "ws.chat.done",
        {
            "thread_id": thread_id,
            "graph": graph_kind,
            "response": response_text,
            "token_count": len(full_response),
        },
    )


def _pick_graph(message: str) -> GraphKind:
    """Choose agent pattern in the background (hidden from the UI)."""
    from agents.graphs.daily import _domains_for

    domains = _domains_for(message)
    if len(domains) > 1 and "general" not in domains:
        return "map_reduce"
    return "simple"


def _validate_chat_payload(data: dict[str, Any]) -> tuple[str, GraphKind] | str:
    message = data.get("message", "")
    if not isinstance(message, str) or not message.strip():
        return "message is required"
    if len(message) > 8000:
        return "message too long"

    graph = data.get("graph")
    if graph is None:
        graph = _pick_graph(message.strip())
    if graph not in ("simple", "map_reduce"):
        return "invalid graph"
    return message.strip(), graph  # type: ignore[return-value]


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    thread_id: str = Query(..., min_length=1, max_length=128),
):
    settings = get_settings()
    if not settings.openai_api_key:
        await websocket.close(code=1013, reason="OPENAI_API_KEY is not configured")
        return

    manager = get_ws_manager()
    set_request_id()
    await manager.connect(websocket, thread_id)
    log_payload(logger, "ws.connect", {"thread_id": thread_id})

    try:
        await websocket.send_json({"type": "connected", "thread_id": thread_id})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "chat")
            log_payload(
                logger,
                "ws.message.in",
                data,
                level=ws_log_level(msg_type, settings=settings),
                thread_id=thread_id,
            )

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "chat":
                await websocket.send_json(
                    {"type": "error", "message": f"unknown message type: {msg_type}"}
                )
                continue

            validated = _validate_chat_payload(data)
            if isinstance(validated, str):
                await websocket.send_json({"type": "error", "message": validated})
                continue

            message, graph_kind = validated
            asyncio.create_task(
                _stream_chat(
                    thread_id=thread_id,
                    message=message,
                    graph_kind=graph_kind,
                )
            )
    except WebSocketDisconnect:
        log_payload(logger, "ws.disconnect", {"thread_id": thread_id})
    finally:
        await manager.disconnect(websocket, thread_id)
