import asyncio
from collections.abc import Coroutine
from typing import Any, Final, TypeVar, Union

from flask import Flask, Response, jsonify, request

from app.llm_model import QwenModel
from memory.convention import MemoryManager
from memory.in_memory import InMemoryMemoryRepository
from memory.llm_ability import LlmAbility
from memory.memory_manager import MemoryManagerImpl
from memory.model import Memory, TextChatMessage

T = TypeVar('T')

# Create a separate LLM model instance for direct operations
llm_model = QwenModel()

# Create the memory manager with its own LLM ability
memory_manager: MemoryManager = MemoryManagerImpl(
    memory_repository=InMemoryMemoryRepository(),
    visible_memories=[],
    llm_ability=LlmAbility(llm_model)
)
app: Final[Flask] = Flask(__name__)

def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Helper function to run async functions in Flask routes."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route("/health", methods=["GET"])
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "message": "Memory Manager API is running"}

# Memory Repository Endpoints
@app.route("/memories", methods=["GET"])
def get_all_memory_abstracts() -> Union[Response, tuple[Response, int]]:
    """Get all memory abstracts from the repository."""
    try:
        abstracts = run_async(memory_manager.memory_repository.fetch_all_memories_abstract())
        return jsonify({"memories":[{"name": abstract.name, "abstract": abstract.abstract} for abstract in abstracts]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memories/<name>", methods=["GET"])
def get_memory_by_name(name: str) -> Union[Response, tuple[Response, int]]:
    """Get a specific memory by name."""
    try:
        memory = run_async(memory_manager.memory_repository.fetch_memory_by_name(name))
        if memory is None:
            return jsonify({"error": f"Memory with name '{name}' not found"}), 404
        return jsonify({
            "name": memory.name,
            "abstract": memory.abstract,
            "memory_block": memory.memory_block
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memories", methods=["POST"])
def add_memory() -> Union[Response, tuple[Response, int]]:
    """Add a new memory to the repository."""
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ["name", "abstract", "memory_block"]):
            return jsonify({"error": "Missing required fields: name, abstract, memory_block"}), 400

        memory = Memory(
            name=data["name"],
            abstract=data["abstract"],
            memory_block=data["memory_block"]
        )

        global memory_manager
        memory_manager = run_async(memory_manager.force_add_memory(memory))
        return jsonify({"message": f"Memory '{memory.name}' added successfully"}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memories/<name>", methods=["PUT"])
def update_memory_by_name(name: str) -> Union[Response, tuple[Response, int]]:
    """Update an existing memory."""
    global memory_manager
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ["abstract", "memory_block"]):
            return jsonify({"error": "Missing required fields: abstract, memory_block"}), 400

        # Check if memory exists
        existing_memory = run_async(memory_manager.memory_repository.fetch_memory_by_name(name))
        if existing_memory is None:
            return jsonify({"error": f"Memory with name '{name}' not found"}), 404

        updated_memory = Memory(
            name=name,
            abstract=data["abstract"],
            memory_block=data["memory_block"]
        )

        memory_manager = run_async(memory_manager.force_update_memory(updated_memory))
        return jsonify({"message": f"Memory '{name}' updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memories/<name>", methods=["DELETE"])
def delete_memory(name: str) -> Union[Response, tuple[Response, int]]:
    """Delete a memory from the repository."""
    try:
        # Check if memory exists
        existing_memory = run_async(memory_manager.memory_repository.fetch_memory_by_name(name))
        if existing_memory is None:
            return jsonify({"error": f"Memory with name '{name}' not found"}), 404

        run_async(memory_manager.memory_repository.remove_memory(existing_memory))
        return jsonify({"message": f"Memory '{name}' deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Memory Manager Endpoints
@app.route("/visible-memories", methods=["GET"])
def get_visible_memories() -> Union[Response, tuple[Response, int]]:
    """Get all visible memories in the current manager."""
    try:
        limit = request.args.get('limit', type=int)
        global memory_manager
        if limit:
            memory_manager = run_async(memory_manager.refresh_visible_memory_list(limit))

        return jsonify({"visible_memories": [{
            "name": memory.name,
            "abstract": memory.abstract,
            "memory_block": memory.memory_block
        } for memory in memory_manager.visible_memories]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/relevance-map", methods=["GET"])
def get_relevance_map() -> Response:
    """Get the current relevance map."""
    return jsonify({"relevance_map":memory_manager.relevance_map})

@app.route("/full-update", methods=["POST"])
def full_update() -> Union[Response, tuple[Response, int]]:
    """Perform a full update cycle: create new memories, update existing ones, and update relevance."""
    try:
        data = request.get_json()
        if not data or "chat_messages" not in data or "delta" not in data:
            return jsonify({"error": "Missing required fields: chat_messages, delta"}), 400

        chat_messages = [
            TextChatMessage(role=msg["role"], text=msg["text"])
            for msg in data["chat_messages"]
        ]
        delta = data["delta"]

        global memory_manager
        memory_manager = run_async(memory_manager.full_update(chat_messages, delta))

        return jsonify({
            "message": "Full update completed successfully",
            "visible_memories": [{
                "name": memory.name,
                "abstract": memory.abstract,
                "memory_block": memory.memory_block
            } for memory in memory_manager.visible_memories],
            "relevance_map": memory_manager.relevance_map
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-existing-memories", methods=["POST"])
def update_existing_memories() -> Union[Response, tuple[Response, int]]:
    """Update all memories in the manager using LLM capabilities."""
    try:
        data = request.get_json()
        if not data or "chat_messages" not in data:
            return jsonify({"error": "Missing required field: chat_messages"}), 400

        chat_messages = [
            TextChatMessage(role=msg["role"], text=msg["text"])
            for msg in data["chat_messages"]
        ]

        global memory_manager
        memory_manager = run_async(memory_manager.update_existing_memories(chat_messages))

        return jsonify({
            "message": "All memories updated successfully",
            "visible_memories": [{
                "name": memory.name,
                "abstract": memory.abstract,
                "memory_block": memory.memory_block
            } for memory in memory_manager.visible_memories]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract-new-memories", methods=["POST"])
def extract_new_memories() -> Union[Response, tuple[Response, int]]:
    """Create new memories based on chat messages and existing memories."""
    try:
        data = request.get_json()
        if not data or "chat_messages" not in data:
            return jsonify({"error": "Missing required field: chat_messages"}), 400

        chat_messages = [
            TextChatMessage(role=msg["role"], text=msg["text"])
            for msg in data["chat_messages"]
        ]

        global memory_manager
        old_count = len(memory_manager.visible_memories)
        memory_manager = run_async(memory_manager.extract_new_memories(chat_messages))
        new_count = len(memory_manager.visible_memories)

        return jsonify({
            "message": f"Created {new_count - old_count} new memories",
            "memories": [{
                "name": memory.name,
                "abstract": memory.abstract,
                "memory_block": memory.memory_block
            } for memory in memory_manager.visible_memories]
        })
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/refresh-visible-memory-list", methods=["POST"])
def refresh_visible_memory_list() -> Union[Response, tuple[Response, int]]:
    """Refresh visible memories based on relevance counts."""
    try:
        data = request.get_json()
        if not data or "limit" not in data:
            return jsonify({"error": "Missing required field: limit"}), 400

        limit = data["limit"]

        global memory_manager
        memory_manager = run_async(memory_manager.refresh_visible_memory_list(limit))

        return jsonify({
            "message": f"Refreshed visible memories list with top {limit} memories",
            "visible_memories": [{
                "name": memory.name,
                "abstract": memory.abstract,
                "memory_block": memory.memory_block
            } for memory in memory_manager.visible_memories]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-relevance-map", methods=["POST"])
def update_relevance_map() -> Union[Response, tuple[Response, int]]:
    """Update relevance map based on memories related to chat messages."""
    try:
        data = request.get_json()
        if not data or "chat_messages" not in data or "delta" not in data:
            return jsonify({"error": "Missing required fields: chat_messages, delta"}), 400

        chat_messages = [
            TextChatMessage(role=msg["role"], text=msg["text"])
            for msg in data["chat_messages"]
        ]
        delta = data["delta"]

        global memory_manager
        memory_manager = run_async(memory_manager.update_relevance_map(chat_messages, delta))

        return jsonify({
            "message": "Relevance map updated successfully",
            "relevance_map": memory_manager.relevance_map
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-visible-memory-list", methods=["POST"])
def update_visible_memory_list() -> Union[Response, tuple[Response, int]]:
    """Update visible memories based on relevance to chat messages."""
    try:
        data = request.get_json()
        if not data or "chat_messages" not in data or "limit" not in data:
            return jsonify({"error": "Missing required fields: chat_messages, limit"}), 400

        chat_messages = [
            TextChatMessage(role=msg["role"], text=msg["text"])
            for msg in data["chat_messages"]
        ]
        limit = data["limit"]
        delta = data.get("delta", 1)  # Default delta value is 1

        global memory_manager
        memory_manager = run_async(memory_manager.update_visible_memory_list(chat_messages, limit, delta))

        return jsonify({
            "message": f"Updated visible memories to top {limit} most relevant",
            "relevance_map": memory_manager.relevance_map,
            "visible_memories": [{
                "name": memory.name,
                "abstract": memory.abstract,
                "memory_block": memory.memory_block
            } for memory in memory_manager.visible_memories]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/force-update-relevance-map", methods=["POST"])
def force_update_relevance_map() -> Union[Response, tuple[Response, int]]:
    """Update the relevance map with delta values."""
    try:
        data = request.get_json()
        if not data or "delta_map" not in data:
            return jsonify({"error": "Missing required field: delta_map"}), 400

        delta_map = data["delta_map"]

        global memory_manager
        memory_manager = run_async(memory_manager.force_update_relevance_map(delta_map))

        return jsonify({
            "message": "Relevance map updated successfully",
            "relevance_map": memory_manager.relevance_map
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_response() -> Union[Response, tuple[Response, int]]:
    """Generate a response using the LLM model."""
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return jsonify({"error": "Missing required field: messages"}), 400

        # Validate messages format
        if not isinstance(data["messages"], list) or not data["messages"]:
            return jsonify({"error": "Messages must be a non-empty list"}), 400

        # Convert to TextChatMessage objects
        try:
            chat_messages = [
                TextChatMessage(role=msg["role"], text=msg["text"])
                for msg in data["messages"]
            ]
        except KeyError as e:
            return jsonify({"error": f"Missing required field in message: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid message format: {str(e)}"}), 400

        # Get optional reasoning parameter (default: True)
        reasoning = data.get("reasoning", True)
        if not isinstance(reasoning, bool):
            return jsonify({"error": "Reasoning parameter must be a boolean"}), 400

        # Generate response using the LLM model
        response = run_async(
            llm_model.generate(chat_messages, reasoning=reasoning)
        )

        return jsonify({
            "response": response,
            "reasoning_enabled": reasoning
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error: Any) -> tuple[Response, int]:
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error: Any) -> tuple[Response, int]:
    print(str(error))
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)