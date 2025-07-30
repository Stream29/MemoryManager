import sys
from typing import Any, TypeVar

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Add the parent directory to the Python path to resolve module imports
sys.path.append("..")

from memory_common.convention import MemorySession
from memory_common.model import Memory, TextChatMessage
from memory_server.in_memory import InMemoryMemoryRepository
from memory_server.llm_ability import LlmAbility
from memory_server.server_memory_session import ServerMemorySession
from server.llm_model import QwenModel

T = TypeVar('T')

# Create a separate LLM model instance for direct operations
llm_model = QwenModel()

# Create the memory_common manager with its own LLM ability
memory_session: MemorySession = ServerMemorySession(
    memory_repository=InMemoryMemoryRepository(),
    llm_ability=LlmAbility(llm_model)
)

app = FastAPI(
    title="Memory Manager API",
    description="API for managing memory and LLM operations",
    version="0.0.1"
)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    print(f"Error processing request {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# MemorySession endpoints
@app.post("/memory", status_code=200)
async def add_memory(request: Request) -> JSONResponse:
    """Endpoint for force_add_memory method."""
    try:
        data = await request.json()
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request data")

        memory_data = data.get("memory")
        if not memory_data:
            raise HTTPException(status_code=400, detail="Memory data is required")

        memory = Memory(**memory_data)
        await memory_session.force_add_memory(memory)
        return JSONResponse(content={"status": "success"}, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in add_memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memory", status_code=200)
async def update_memory_by_object(request: Request) -> JSONResponse:
    """Endpoint for force_update_memory method."""
    try:
        data = await request.json()
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request data")

        memory_data = data.get("memory")
        if not memory_data:
            raise HTTPException(status_code=400, detail="Memory data is required")

        memory = Memory(**memory_data)
        await memory_session.force_update_memory(memory)
        return JSONResponse(content={"status": "success"}, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in update_memory_by_object: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{name}", status_code=200)
async def remove_memory(name: str) -> JSONResponse:
    """Endpoint for force_remove_memory_by_name method."""
    try:
        await memory_session.force_remove_memory_by_name(name)
        return JSONResponse(content={"status": "success"}, status_code=200)
    except Exception as e:
        print(f"Error in remove_memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory-from-chat", status_code=200)
async def update_memory_from_chat(request: Request) -> JSONResponse:
    """Endpoint for update_memory method."""
    try:
        data = await request.json()
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request data")

        chat_messages_data = data.get("chat_messages")
        if not chat_messages_data or not isinstance(chat_messages_data, list):
            raise HTTPException(status_code=400, detail="Chat messages are required")

        chat_messages = [TextChatMessage(**msg) for msg in chat_messages_data]
        await memory_session.update_memory(chat_messages)
        return JSONResponse(content={"status": "success"}, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in update_memory_from_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory-context", status_code=200)
async def get_context_memories() -> JSONResponse:
    """Endpoint for retrieve_context_memories method."""
    try:
        memories = await memory_session.retrieve_context_memories()
        return JSONResponse(content={"memories": [memory.model_dump() for memory in memories]}, status_code=200)
    except Exception as e:
        print(f"Error in get_context_memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{name}", status_code=200)
async def get_memory(name: str) -> JSONResponse:
    """Endpoint for fetch_memory_by_name method."""
    try:
        memory = await memory_session.fetch_memory_by_name(name)
        if memory is None:
            raise HTTPException(status_code=404, detail=f"Memory with name '{name}' not found")
        return JSONResponse(content={"memory": memory.model_dump()}, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory-abstracts", status_code=200)
async def get_all_memory_abstracts() -> JSONResponse:
    """Endpoint for fetch_all_memories_abstract method."""
    try:
        abstracts = await memory_session.fetch_all_memories_abstract()
        return JSONResponse(content={"abstracts": [abstract.model_dump() for abstract in abstracts]}, status_code=200)
    except Exception as e:
        print(f"Error in get_all_memory_abstracts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# LlmModel endpoints
@app.post("/llm/generate", status_code=200)
async def generate_text(request: Request) -> JSONResponse:
    """Endpoint for LlmModel's generate method."""
    try:
        data = await request.json()
        if not data or not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request data")

        messages_data = data.get("messages")
        if not messages_data or not isinstance(messages_data, list):
            raise HTTPException(status_code=400, detail="Messages are required")

        reasoning = data.get("reasoning", True)

        messages = [TextChatMessage(**msg) for msg in messages_data]
        result = await llm_model.generate(messages, reasoning)
        return JSONResponse(content={"generated_text": result}, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)