from asyncio import gather
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
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
    
    The MemoryManager provides two types of operations:
    1. Basic operations: force_add_memory, force_update_memory, force_update_relevance_map
    2. High-level operations: Operations that combine basic operations with LLM capabilities
    
    Attributes:
        memory_repository: Repository for persistent memory storage
        visible_memories: Memories visible in this manager
        _llm_ability: LLM capabilities for memory processing
        relevance_map: Mapping of memory names to their relevance counts
    """
    memory_repository: Final[MemoryRepository]
    visible_memories: Final[Sequence[Memory]]
    _llm_ability: Final[LlmAbility]
    relevance_map: Final[Mapping[str, int]]

    def __init__(
            self,
            memory_repository: MemoryRepository,
            visible_memories: Sequence[Memory],
            llm_ability: LlmAbility,
            relevance_map: Mapping[str, int] | None = None
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
        self._llm_ability = llm_ability
        self.relevance_map = relevance_map or {}

    async def force_add_memory(self, memory: Memory) -> "MemoryManager":
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
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def force_update_relevance_map(self, delta_map: Mapping[str, int]) -> "MemoryManager":
        """
        Update the relevance map with delta values.
        
        Args:
            delta_map: Mapping of memory names to delta relevance values to add
            
        Returns:
            New MemoryManager instance with updated relevance map
        """
        new_relevance_map: Final[MutableMapping[str, int]] = {**self.relevance_map, **delta_map}
        for name, delta_value in delta_map.items():
            if name in new_relevance_map:
                new_relevance_map[name] += delta_value
            else:
                new_relevance_map[name] = delta_value
        return MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=self.visible_memories,
            llm_ability=self._llm_ability,
            relevance_map=new_relevance_map
        )

    async def force_update_memory(self, memory: Memory) -> "MemoryManager":
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
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def full_update(
            self,
            chat_messages: Sequence[TextChatMessage],
            delta: int
    ) -> "MemoryManager":
        """
        Perform a full update cycle: create new memories, update existing ones, and update relevance.
        
        Args:
            chat_messages: Chat messages to analyze for memory operations
            delta: The delta value to add to relevance counts of related memories
            
        Returns:
            New MemoryManager instance with all updates applied
        """
        current_memory_list: Final[
            Sequence[MemoryAbstract]] = await self.memory_repository.fetch_all_memories_abstract()
        new_memories, related_memories, updated_memories = await gather(
            self._llm_ability.extract_new_memories(
                current_memories=current_memory_list,
                chat_messages=chat_messages
            ),
            self._llm_ability.list_related_memories(
                current_memories=current_memory_list,
                chat_messages=chat_messages
            ),
            self._get_updated_memories(current_memory_list, chat_messages)
        )
        new_memory_manager = self
        for new_memory in new_memories:
            try:
                new_memory_manager = await new_memory_manager.force_add_memory(new_memory)
            except ValueError as e:
                print(e)
        for updated_memory in updated_memories:
            try:
                new_memory_manager = await new_memory_manager.force_update_memory(updated_memory)
            except ValueError as e:
                print(e)
        new_memory_manager = await new_memory_manager.force_update_relevance_map({
            memory.name: delta for memory in related_memories
        })
        return new_memory_manager

    async def update_existing_memories(
            self,
            chat_messages: Sequence[TextChatMessage]
    ) -> "MemoryManager":
        """
        Update all memories in the manager using LLM capabilities.
        
        Delegates to the LLM ability to analyze chat history and update
        all relevant memories based on new information.
        
        Args:
            chat_messages: Chat messages to analyze for memory updates
        
        Returns:
            New MemoryManager instance with updated memories
        """
        updated_memories: Final[Sequence[Memory]] = await self._get_updated_memories(
            current_memories=await self.memory_repository.fetch_all_memories_abstract(),
            chat_messages=chat_messages,
        )

        # Apply all updates to create new memory manager
        new_memory_manager = self
        for memory in updated_memories:
            new_memory_manager = await new_memory_manager.force_update_memory(memory)
        return new_memory_manager

    async def extract_new_memories(
            self,
            chat_messages: Sequence[TextChatMessage]
    ) -> "MemoryManager":
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
        new_memories = await self._llm_ability.extract_new_memories(
            await self.memory_repository.fetch_all_memories_abstract(),
            chat_messages
        )

        # Add all new memories to the manager
        new_manager = self
        for memory in new_memories:
            new_manager = await new_manager.force_add_memory(memory)

        return new_manager

    async def refresh_visible_memory_list(self, limit: int) -> "MemoryManager":
        """
        Refresh visible memories based on relevance counts.
        
        Selects the top n most relevant memories based on the current
        relevance map to be visible in the new MemoryManager.
        
        Args:
            limit: Number of most relevant memories to keep visible

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
        top_memories = sorted_memories[:limit]

        # Fetch full memory objects for visible memories
        visible_memories = []
        for memory_abstract in top_memories:
            memory = await self.memory_repository.fetch_memory_by_name(memory_abstract.name)
            if memory:
                visible_memories.append(memory)

        return MemoryManager(
            memory_repository=self.memory_repository,
            visible_memories=visible_memories,
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def update_relevance_map(self, chat_messages: Sequence[TextChatMessage], delta: int) -> "MemoryManager":
        """
        Update relevance map based on memories related to chat messages.
        
        Args:
            chat_messages: Chat messages to analyze for related memories
            delta: The delta value to add to relevance counts
            
        Returns:
            New MemoryManager instance with updated relevance map
        """
        related_memories_list: Final[Sequence[MemoryAbstract]] = \
            await self._llm_ability.list_related_memories(
                await self.memory_repository.fetch_all_memories_abstract(),
                chat_messages
            )

        delta_map: Final[Mapping[str, int]] = {
            related_memory.name: delta for related_memory in related_memories_list
        }

        return await self.force_update_relevance_map(delta_map)

    async def update_visible_memory_list(
            self,
            chat_messages: Sequence[TextChatMessage],
            limit: int,
            delta: int = 1
    ) -> "MemoryManager":
        """
        Update visible memories based on relevance to chat messages.
        
        Uses LLM to find memories associated with the given chat messages,
        updates relevance counts, and selects the top n most relevant
        memories to be visible in the new MemoryManager.
        
        Args:
            chat_messages: Chat messages to find memory associations with
            limit: Number of most relevant memories to keep visible
            delta: The delta value to add to relevance counts of related memories
            
        Returns:
            New MemoryManager instance with top n most relevant memories
        """
        return await (await self.update_relevance_map(chat_messages, delta)).refresh_visible_memory_list(limit)

    async def _get_updated_memory(
            self,
            memory_abstract: MemoryAbstract,
            chat_messages: Sequence[TextChatMessage]
    ) -> Memory:
        """
        Get an updated memory based on chat messages.
        
        Args:
            memory_abstract: The abstract of the memory to update
            chat_messages: Chat messages to analyze for memory updates
            
        Returns:
            Updated Memory object
            
        Raises:
            ValueError: If the memory with the given abstract does not exist
        """
        old_memory: Final[Memory | None] = await self.memory_repository.fetch_memory_by_name(memory_abstract.name)
        if old_memory is None:
            raise ValueError(f"Memory with abstract {memory_abstract.model_dump_json()} does not exist")
        return await self._llm_ability.update_memory(
            old_memory=old_memory,
            chat_messages=chat_messages
        )

    async def _get_updated_memories(
            self,
            current_memories: Sequence[MemoryAbstract],
            chat_messages: Sequence[TextChatMessage]
    ) -> Sequence[Memory]:
        """
        Get updated memories based on chat messages.
        
        Args:
            current_memories: Current memory abstracts to consider for updates
            chat_messages: Chat messages to analyze for memory updates
            
        Returns:
            Sequence of updated Memory objects
        """
        memories_to_update: Final[Sequence[MemoryAbstract]] = await self._llm_ability.list_memory_to_update(
            current_memories,
            chat_messages
        )

        # Concurrently update all identified memories
        updated_memories: Final[Sequence[Memory]] = await gather(
            *[self._get_updated_memory(memory_abstract=memory_abstract, chat_messages=chat_messages) for
              memory_abstract in memories_to_update]
        )
        return updated_memories