from typing import Any, Final, TypeVar

from flask import Flask, Response, jsonify

from memory_common.convention import MemorySession
from memory_server.in_memory import InMemoryMemoryRepository
from memory_server.llm_ability import LlmAbility
from memory_server.server_memory_session import ServerMemorySession
from server.llm_model import QwenModel

T = TypeVar('T')

# Create a separate LLM model instance for direct operations
llm_model = QwenModel()

# Create the memory_common manager with its own LLM ability
memory_manager: MemorySession = ServerMemorySession(
    memory_repository=InMemoryMemoryRepository(),
    visible_memories=[],
    llm_ability=LlmAbility(llm_model)
)
app: Final[Flask] = Flask(__name__)


@app.errorhandler(500)
def internal_error(error: Any) -> tuple[Response, int]:
    print(str(error))
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
