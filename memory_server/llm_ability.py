import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeVar, final

from pydantic import BaseModel

from memory_common.convention import LlmModel

if TYPE_CHECKING:
    pass
from memory_common.model import (
    CreateNewMemoriesRequest,
    CreateNewMemoriesResponse,
    Memory,
    MemoryAbstract,
    TextChatMessage,
    UpdateMemoriesRequest,
    UpdateMemoriesResponse,
    UpdateSingleMemoryRequest,
    UpdateSingleMemoryResponse,
)
from memory_server.prompt import (
    new_memory_system_prompt,
    update_memories_system_prompt,
    update_single_memory_system_prompt,
)

T = TypeVar("T", bound=BaseModel)


@final
@dataclass
class LlmAbility:
    """
    Provides LLM-powered capabilities for memory_common management operations.
    
    This class handles the integration with Large Language Models to perform
    intelligent memory_common updates based on chat history and existing memories.
    It provides structured generation capabilities and memory_common update logic.
    
    The LlmAbility provides stateless functions for processing memory_common data:
    1. Memory Updating Functions: update_memory, list_memory_to_update
    2. Memory Creation Functions: extract_new_memories
    3. Memory Association Functions: list_related_memories
    
    Attributes:
        _llm_model: The LLM model instance used for generation
    """
    _llm_model: Final[LlmModel]

    def __init__(self, llm_model: LlmModel):
        """
        Initialize LlmAbility with a specific LLM model.
        
        Args:
            llm_model: The LLM model to use for memory_common operations
        """
        self._llm_model = llm_model

    @staticmethod
    def _safe_cast(target_type: type[T], value: str) -> T:
        """
        Safely extract and parse JSON from LLM response string.
        
        Extracts JSON content from potentially noisy LLM responses using regex,
        then validates and converts it to the specified Pydantic model type.
        
        Args:
            target_type: The Pydantic model type to convert to
            value: The raw string response from the LLM
            
        Returns:
            Validated instance of the target type
            
        Raises:
            ValueError: If no valid JSON is found in the response
            ValidationError: If the JSON doesn't match the target schema
        """
        # Extract JSON content using regex (handles extra text around JSON)
        match: Final[re.Match[str] | None] = re.search(r'\{.*}', value, re.DOTALL)
        if match is None:
            raise ValueError(f"Invalid JSON string: {value}")

        safe_string: Final[str] = match.group(0)
        json_data = json.loads(safe_string)
        return target_type.model_validate(json_data)

    async def _structured_generate(self, request: BaseModel, system_prompt: str, response_type: type[T]) -> T:
        """
        Generate a structured response using the LLM with a specific format.
        
        Sends a system prompt and request data to the LLM, then parses the response
        into the specified Pydantic model type for structured data handling.
        
        Args:
            request: The request data to send to the LLM (as JSON)
            system_prompt: System prompt defining the task and response format
            response_type: Expected Pydantic model type for the response
            
        Returns:
            Parsed and validated response of the specified type
            
        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
            ValidationError: If the response doesn't match the expected schema
        """
        response_str: Final[str] = await self._llm_model.generate(
            messages=[
                TextChatMessage(role="system", text=system_prompt),
                TextChatMessage(role="user", text=request.model_dump_json())
            ]
        )
        return LlmAbility._safe_cast(response_type, response_str)

    async def update_memory(
            self,
            old_memory: Memory,
            chat_messages: Sequence[TextChatMessage]
    ) -> Memory:
        """
        Update a single memory_common based on chat history.
        
        Args:
            old_memory: The memory_common to update
            chat_messages: Chat messages to analyze for memory_common updates
            
        Returns:
            Updated Memory object with new content
        """
        # Create request for updating single memory_common
        request = UpdateSingleMemoryRequest(
            chat_history=chat_messages,
            old_memory=old_memory
        )

        # Generate updated memory_common block using LLM
        response: Final[UpdateSingleMemoryResponse] = await self._structured_generate(
            request,
            update_single_memory_system_prompt,
            UpdateSingleMemoryResponse
        )

        # Return updated Memory object with new content
        return Memory(
            name=old_memory.name,
            abstract=old_memory.abstract,
            memory_block=response.new_memory_block
        )

    async def list_memory_to_update(
            self,
            current_memory: Sequence[MemoryAbstract],
            chat_messages: Sequence[TextChatMessage]
    ) -> Sequence[MemoryAbstract]:
        """
        Determine which memories should be updated based on chat messages.
        
        Args:
            current_memory: Current memory_common abstracts to consider for updates
            chat_messages: Chat messages to analyze for memory_common updates
            
        Returns:
            Sequence of MemoryAbstract objects that should be updated
        """
        # Create request to determine which memories need updating
        request = UpdateMemoriesRequest(
            chat_history=chat_messages,
            old_memory=current_memory,
        )

        # Get list of memory_common names that need updating
        response: Final[UpdateMemoriesResponse] = await self._structured_generate(
            request,
            update_memories_system_prompt,
            UpdateMemoriesResponse
        )

        return [memory for memory in current_memory if memory.name in response.memories_to_update]

    async def extract_new_memories(
            self,
            current_memories: Sequence[MemoryAbstract],
            chat_messages: Sequence[TextChatMessage]
    ) -> Sequence[Memory]:
        """
        Create new memories based on chat messages and existing memories.
        
        Analyzes the chat messages to identify information that is not covered
        by existing memories and creates new memory_common blocks for that information.
        
        Args:
            current_memories: Current memory_common abstracts to consider
            chat_messages: Chat messages to analyze for new memory_common creation
            
        Returns:
            Sequence of new Memory objects that should be created
        """
        # Create request for new memory_common creation
        request = CreateNewMemoriesRequest(
            current_memories=current_memories,
            chat_history=chat_messages
        )

        # Generate new memories using LLM
        response: Final[CreateNewMemoriesResponse] = await self._structured_generate(
            request,
            new_memory_system_prompt,
            CreateNewMemoriesResponse
        )

        return response.new_memories
