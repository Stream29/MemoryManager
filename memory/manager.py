from collections.abc import MutableSequence, Sequence
from typing import Final, final

from memory.convention import MemoryRepository
from memory.llm_ability import LlmAbility
from memory.model import Memory, MemoryAbstract, TextChatMessage


@final
class MemoryManager:
    """
    Represents a manager containing memories and LLM capabilities.
    
    A MemoryManager encapsulates the context needed for memory operations,
    including the storage backend, visible memories, and LLM abilities 
    for processing updates.
    
    Attributes:
        memory_repository: Repository for persistent memory storage
        visible_memories: Memories visible in this manager
        llm_ability: LLM capabilities for memory processing
        relevance_map: Mapping of memory names to their relevance counts
    """
    memory_repository: Final[MemoryRepository]
    visible_memories: Final[Sequence[Memory]]
    llm_ability: Final[LlmAbility]
    relevance_map: Final[dict[str, int]]

    def __init__(
            self,
            memory_repository: MemoryRepository,
            visible_memories: Sequence[Memory],
            llm_ability: LlmAbility,
            relevance_map: dict[str, int] | None = None
    ):
        """
        Initialize a new MemoryManager.
        
        Args:
            memory_repository: Repository for persistent memory storage
            visible_memories: Memories visible in this manager
            llm_ability: LLM capabilities for memory processing
            relevance_map: Mapping of memory names to their relevance counts
        """
        self.memory_repository = memory_repository
        self.visible_memories = visible_memories
        self.llm_ability = llm_ability
        self.relevance_map = relevance_map or {}

    async def add_memory(self, memory: Memory) -> "MemoryManager":
        """
        Add a new memory to the manager.
        
        Creates a new MemoryManager with the added memory included in
        both the storage and visible memories.
        
        Args:
            memory: The memory to add
            
        Returns:
            New MemoryManager instance with the added memory
            
        Raises:
            ValueError: If a memory with the same name already exists
        """
        # Check if memory with same name already exists
        if any(x.name == memory.name for x in self.visible_memories):
            raise ValueError(f"Memory with name {memory.name} already exists")

        await self.memory_repository.add_memory(memory)
        return MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=[*self.visible_memories, memory],
            llm_ability=self.llm_ability,
            relevance_map=self.relevance_map
        )

    async def update_all_memories(self, chat_messages: Sequence[TextChatMessage]) -> "MemoryManager":
        """
        Update all memories in the manager using LLM capabilities.
        
        Delegates to the LLM ability to analyze chat history and update
        all relevant memories based on new information.
        
        Args:
            chat_messages: Chat messages to analyze for memory updates
        
        Returns:
            New MemoryManager instance with updated memories
        """
        return await self.llm_ability.update_all_memories(self, chat_messages)

    async def create_new_memories(self, chat_messages: Sequence[TextChatMessage]) -> "MemoryManager":
        """
        Create new memories based on chat messages and existing memories.
        
        Delegates to the LLM ability to analyze chat messages and identify
        information not covered by existing memories, then creates new
        memory blocks and adds them to the manager.
        
        Args:
            chat_messages: Chat messages to analyze for new memory creation
        
        Returns:
            New MemoryManager instance with newly created memories added
        """
        new_memories = await self.llm_ability.create_new_memories(self, chat_messages)

        # Add all new memories to the manager
        new_manager = self
        for memory in new_memories:
            new_manager = await new_manager.add_memory(memory)

        return new_manager

    async def update_memory(self, memory: Memory) -> "MemoryManager":
        """
        Update a specific memory in the manager.
        
        Replaces the existing memory with the updated version in both
        the visible memories and persistent storage.
        
        Args:
            memory: The updated memory to replace the existing one
            
        Returns:
            New MemoryManager instance with the updated memory
        """
        new_visible_memories: MutableSequence[Memory] = [*self.visible_memories]
        new_visible_memories.remove(memory)
        new_visible_memories.append(memory)
        await self.memory_repository.update_memory(memory)
        return MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=new_visible_memories,
            llm_ability=self.llm_ability,
            relevance_map=self.relevance_map
        )

    async def refresh_visible_memories(self, n: int) -> "MemoryManager":
        """
        Refresh visible memories based on relevance counts.
        
        Selects the top n most relevant memories based on the current
        relevance map to be visible in the new MemoryManager.
        
        Args:
            n: Number of most relevant memories to keep visible

        Returns:
            New MemoryManager instance with updated visible memories
        """
        # Get all memories from repository
        all_memories = await self.memory_repository.fetch_all_memories_abstract()

        # Sort memories by relevance count and get top n
        sorted_memories = sorted(
            all_memories,
            key=lambda x: self.relevance_map.get(x.name, 0),
            reverse=True
        )
        top_memories = sorted_memories[:n]

        # Fetch full memory objects for visible memories
        visible_memories = []
        for memory_abstract in top_memories:
            memory = await self.memory_repository.fetch_memory_by_name(memory_abstract.name)
            if memory:
                visible_memories.append(memory)

        return MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=visible_memories,
            llm_ability=self.llm_ability,
            relevance_map=self.relevance_map
        )

    async def update_visible_memories(self, chat_messages: Sequence[TextChatMessage], n: int) -> "MemoryManager":
        """
        Update visible memories based on relevance to chat messages.
        
        Uses LLM to find memories associated with the given chat messages,
        counts relevance occurrences, and selects the top n most relevant
        memories to be visible in the new MemoryManager.
        
        Args:
            chat_messages: Chat messages to find memory associations with
            n: Number of most relevant memories to keep visible
            
        Returns:
            New MemoryManager instance with top n most relevant memories
        """
        # Get all memories from repository
        all_memory_abstracts: Final[
            Sequence[MemoryAbstract]] = await self.memory_repository.fetch_all_memories_abstract()

        new_relevance: Final[Sequence[str]] = await self.llm_ability.find_associated_memories(self, chat_messages)
        new_relevance_map: Final[dict[str, int]] = {**self.relevance_map}

        for name in new_relevance:
            if name in new_relevance_map:
                new_relevance_map[name] += 1
            else:
                new_relevance_map[name] = 1

        new_memory_manager: Final[MemoryManager] = await MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=[],
            llm_ability=self.llm_ability,
            relevance_map=new_relevance_map
        ).refresh_visible_memories(n)

        return new_memory_manager
