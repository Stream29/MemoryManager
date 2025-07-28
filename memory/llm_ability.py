import json
import re
from asyncio import gather
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeVar, final, TYPE_CHECKING

from pydantic import BaseModel

from memory.convention import LlmModel
if TYPE_CHECKING:
    from memory.manager import MemoryManager
from memory.model import (
    CreateNewMemoriesRequest,
    CreateNewMemoriesResponse,
    FindAssociatedMemoriesRequest,
    FindAssociatedMemoriesResponse,
    Memory,
    MemoryAbstract,
    TextChatMessage,
    UpdateMemoriesRequest,
    UpdateMemoriesResponse,
    UpdateSingleMemoryRequest,
    UpdateSingleMemoryResponse,
)
from memory.prompt import (
    find_associated_memories_system_prompt,
    new_memory_system_prompt,
    update_memories_system_prompt,
    update_single_memory_system_prompt,
)

T = TypeVar("T", bound=BaseModel)


@final
@dataclass
class LlmAbility:
    """
    Provides LLM-powered capabilities for memory management operations.
    
    This class handles the integration with Large Language Models to perform
    intelligent memory updates based on chat history and existing memories.
    It provides structured generation capabilities and memory update logic.
    
    Attributes:
        llm_model: The LLM model instance used for generation
    """
    llm_model: Final[LlmModel]

    def __init__(self, llm_model: LlmModel):
        """
        Initialize LlmAbility with a specific LLM model.
        
        Args:
            llm_model: The LLM model to use for memory operations
        """
        self.llm_model = llm_model

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
        response_str: Final[str] = await self.llm_model.generate(
            messages=[
                TextChatMessage(role="system", text=system_prompt),
                TextChatMessage(role="user", text=request.model_dump_json())
            ]
        )
        return LlmAbility._safe_cast(response_type, response_str)

    async def update_memory_by_name(self, memory_scope: "MemoryManager", name: str) -> Memory:
        """
        Update a specific memory by name using LLM analysis.
        
        Finds the memory with the given name in the scope, then uses the LLM
        to generate an updated memory block based on the chat history and
        existing memory content.
        
        Args:
            memory_scope: The memory manager containing the memory to update
            name: The name of the memory to update
            
        Returns:
            Updated Memory object with new content but same name and abstract
            
        Raises:
            ValueError: If no memory with the given name is found
        """
        # Find the memory by name in the visible memories
        old_memory: Memory | None = None
        for memory in memory_scope.visible_memories:
            if memory.name == name:
                old_memory = memory
                break
        
        if old_memory is None:
            raise ValueError(f"Memory with name '{name}' not found in memory manager")
        
        # Create request for updating single memory
        request = UpdateSingleMemoryRequest(
            chat_history=memory_scope.visible_chat_messages,
            old_memory=old_memory
        )
        
        # Generate updated memory block using LLM
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
        

    async def update_all_memories(self, memory_scope: "MemoryManager") -> "MemoryManager":
        """
        Update all relevant memories in the scope based on chat history.
        
        First determines which memories need updating by analyzing chat history
        and existing memory abstracts, then concurrently updates all identified
        memories using LLM analysis.
        
        Args:
            memory_scope: The memory manager containing memories to potentially update
            
        Returns:
            New MemoryManager with all relevant memories updated
        """
        # Create request to determine which memories need updating
        request = UpdateMemoriesRequest(
            chat_history=memory_scope.visible_chat_messages,
            old_memory=[
                MemoryAbstract(
                    name=memory.name,
                    abstract=memory.abstract
                ) for memory in memory_scope.visible_memories
            ],
        )
        
        # Get list of memory names that need updating
        response: Final[UpdateMemoriesResponse] = await self._structured_generate(
            request,
            update_memories_system_prompt,
            UpdateMemoriesResponse
        )
        
        # Concurrently update all identified memories
        updated_memories: Final[Sequence[Memory]] = await gather(
            *[self.update_memory_by_name(memory_scope=memory_scope, name=name) 
              for name in response.memories_to_update]
        )
        
        # Apply all updates to create new memory manager
        new_memory_manager = memory_scope
        for memory in updated_memories:
            new_memory_manager = await new_memory_manager.update_memory(memory)
        return new_memory_manager

    async def create_new_memories(self, memory_scope: "MemoryManager") -> Sequence[Memory]:
        """
        Create new memories based on chat history and existing memories.
        
        Analyzes the chat history to identify information that is not covered
        by existing memories and creates new memory blocks for that information.
        
        Args:
            memory_scope: The memory manager containing existing memories and chat history
            
        Returns:
            Sequence of new Memory objects that should be created
        """
        # Create request for new memory creation
        request = CreateNewMemoriesRequest(
            current_memories=[
                MemoryAbstract(
                    name=memory.name,
                    abstract=memory.abstract
                ) for memory in memory_scope.visible_memories
            ],
            chat_history=memory_scope.visible_chat_messages
        )
        
        # Generate new memories using LLM
        response: Final[CreateNewMemoriesResponse] = await self._structured_generate(
            request,
            new_memory_system_prompt,
            CreateNewMemoriesResponse
        )
        
        return response.new_memories

    async def find_associated_memories(
            self,
            memory_scope: "MemoryManager",
            chat_messages: Sequence[TextChatMessage]
    ) -> Sequence[str]:
        """
        Find memories that are associated with the given chat messages.
        
        Analyzes the chat messages against existing memories to determine
        which memories are most relevant to the current conversation context.
        
        Args:
            memory_scope: The memory manager containing existing memories
            chat_messages: Chat messages to find associations with
            
        Returns:
            Sequence of memory names that are associated with the chat messages
        """
        # Create request for finding associated memories
        request = FindAssociatedMemoriesRequest(
            current_memories=[
                MemoryAbstract(
                    name=memory.name,
                    abstract=memory.abstract
                ) for memory in memory_scope.visible_memories
            ],
            chat_messages=chat_messages
        )
        
        # Find associated memories using LLM
        response: Final[FindAssociatedMemoriesResponse] = await self._structured_generate(
            request,
            find_associated_memories_system_prompt,
            FindAssociatedMemoriesResponse
        )
        
        return response.associated_memories
