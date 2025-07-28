from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

from memory.convention import MemoryRepository
from memory.model import Memory, TextChatMessage

if TYPE_CHECKING:
    from memory.llm_ability import LlmAbility


@final
@dataclass
class MemoryScope:
    """
    Represents a scope containing memories, chat messages, and LLM capabilities.
    
    A MemoryScope encapsulates the context needed for memory operations,
    including the storage backend, visible memories, chat history, and
    LLM abilities for processing updates.
    
    Attributes:
        memory_storage: Repository for persistent memory storage
        visible_chat_messages: Chat messages visible in this scope
        visible_memories: Memories visible in this scope
        llm_ability: LLM capabilities for memory processing
    """
    memory_storage: Final[MemoryRepository]
    visible_chat_messages: Final[Sequence[TextChatMessage]]
    visible_memories: Final[Sequence[Memory]]
    llm_ability: Final["LlmAbility"]
    
    def __init__(
        self,
        memory_storage: MemoryRepository,
        visible_chat_messages: Sequence[TextChatMessage],
        visible_memories: Sequence[Memory],
        llm_ability: "LlmAbility"
    ):
        """
        Initialize a new MemoryScope.
        
        Args:
            memory_storage: Repository for persistent memory storage
            visible_chat_messages: Chat messages visible in this scope
            visible_memories: Memories visible in this scope
            llm_ability: LLM capabilities for memory processing
        """
        self.memory_storage = memory_storage
        self.visible_chat_messages = visible_chat_messages
        self.visible_memories = visible_memories
        self.llm_ability = llm_ability

    async def add_memory(self, memory: Memory) -> "MemoryScope":
        """
        Add a new memory to the scope.
        
        Creates a new MemoryScope with the added memory included in
        both the storage and visible memories.
        
        Args:
            memory: The memory to add
            
        Returns:
            New MemoryScope instance with the added memory
            
        Raises:
            ValueError: If a memory with the same name already exists
        """
        # Check if memory with same name already exists
        if any(x.name == memory.name for x in self.visible_memories):
            raise ValueError(f"Memory with name {memory.name} already exists")
        
        await self.memory_storage.add_memory(memory)
        return MemoryScope(
            memory_storage=self.memory_storage,
            visible_chat_messages=self.visible_chat_messages,
            visible_memories=[*self.visible_memories, memory],
            llm_ability=self.llm_ability
        )

    async def update_all_memories(self) -> "MemoryScope":
        """
        Update all memories in the scope using LLM capabilities.
        
        Delegates to the LLM ability to analyze chat history and update
        all relevant memories based on new information.
        
        Returns:
            New MemoryScope instance with updated memories
        """
        return await self.llm_ability.update_all_memories(self)

    async def create_new_memories(self) -> "MemoryScope":
        """
        Create new memories based on chat history and existing memories.
        
        Delegates to the LLM ability to analyze chat history and identify
        information not covered by existing memories, then creates new
        memory blocks and adds them to the scope.
        
        Returns:
            New MemoryScope instance with newly created memories added
        """
        new_memories = await self.llm_ability.create_new_memories(self)
        
        # Add all new memories to the scope
        new_scope = self
        for memory in new_memories:
            new_scope = await new_scope.add_memory(memory)
        
        return new_scope

    async def update_memory(self, memory: Memory) -> "MemoryScope":
        """
        Update a specific memory in the scope.
        
        Replaces the existing memory with the updated version in both
        the visible memories and persistent storage.
        
        Args:
            memory: The updated memory to replace the existing one
            
        Returns:
            New MemoryScope instance with the updated memory
        """
        new_visible_memories: MutableSequence[Memory] = [*self.visible_memories]
        new_visible_memories.remove(memory)
        new_visible_memories.append(memory)
        await self.memory_storage.update_memory(memory)
        return MemoryScope(
            memory_storage=self.memory_storage,
            visible_chat_messages=self.visible_chat_messages,
            visible_memories=new_visible_memories,
            llm_ability=self.llm_ability
        )