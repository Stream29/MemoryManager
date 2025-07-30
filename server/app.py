from typing import Any, Final, TypeVar

from flask import Flask, Response, jsonify, request

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
app: Final[Flask] = Flask(__name__)


# MemorySession endpoints
@app.route("/memory", methods=["POST"])
async def add_memory() -> tuple[Response, int]:
    """Endpoint for force_add_memory method."""
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request data"}), 400

        memory_data = data.get("memory")
        if not memory_data:
            return jsonify({"error": "Memory data is required"}), 400

        memory = Memory(**memory_data)
        await memory_session.force_add_memory(memory)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error in add_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory", methods=["PUT"])
async def update_memory_by_object() -> tuple[Response, int]:
    """Endpoint for force_update_memory method."""
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request data"}), 400

        memory_data = data.get("memory")
        if not memory_data:
            return jsonify({"error": "Memory data is required"}), 400

        memory = Memory(**memory_data)
        await memory_session.force_update_memory(memory)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error in update_memory_by_object: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory/<name>", methods=["DELETE"])
async def remove_memory(name: str) -> tuple[Response, int]:
    """Endpoint for force_remove_memory_by_name method."""
    try:
        await memory_session.force_remove_memory_by_name(name)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error in remove_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory/from_chat", methods=["POST"])
async def update_memory_from_chat() -> tuple[Response, int]:
    """Endpoint for update_memory method."""
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request data"}), 400

        chat_messages_data = data.get("chat_messages")
        if not chat_messages_data or not isinstance(chat_messages_data, list):
            return jsonify({"error": "Chat messages are required"}), 400

        chat_messages = [TextChatMessage(**msg) for msg in chat_messages_data]
        await memory_session.update_memory(chat_messages)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error in update_memory_from_chat: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory/context", methods=["GET"])
async def get_context_memories() -> tuple[Response, int]:
    """Endpoint for retrieve_context_memories method."""
    try:
        memories = await memory_session.retrieve_context_memories()
        return jsonify({"memories": [memory.model_dump() for memory in memories]}), 200
    except Exception as e:
        print(f"Error in get_context_memories: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory/<name>", methods=["GET"])
async def get_memory(name: str) -> tuple[Response, int]:
    """Endpoint for fetch_memory_by_name method."""
    try:
        memory = await memory_session.fetch_memory_by_name(name)
        if memory is None:
            return jsonify({"error": f"Memory with name '{name}' not found"}), 404
        return jsonify({"memory": memory.model_dump()}), 200
    except Exception as e:
        print(f"Error in get_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory/abstracts", methods=["GET"])
async def get_all_memory_abstracts() -> tuple[Response, int]:
    """Endpoint for fetch_all_memories_abstract method."""
    try:
        abstracts = await memory_session.fetch_all_memories_abstract()
        return jsonify({"abstracts": [abstract.model_dump() for abstract in abstracts]}), 200
    except Exception as e:
        print(f"Error in get_all_memory_abstracts: {str(e)}")
        return jsonify({"error": str(e)}), 500


# LlmModel endpoints
@app.route("/llm/generate", methods=["POST"])
async def generate_text() -> tuple[Response, int]:
    """Endpoint for LlmModel's generate method."""
    try:
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request data"}), 400

        messages_data = data.get("messages")
        if not messages_data or not isinstance(messages_data, list):
            return jsonify({"error": "Messages are required"}), 400

        reasoning = data.get("reasoning", True)

        messages = [TextChatMessage(**msg) for msg in messages_data]
        result = await llm_model.generate(messages, reasoning)
        return jsonify({"generated_text": result}), 200
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(500)
def internal_error(error: Any) -> tuple[Response, int]:
    print(str(error))
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
