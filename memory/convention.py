from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from memory.model import Memory, MemoryAbstract, TextChatMessage


class LlmModel(ABC):
    """
    Abstract base class for Large Language Model implementations.
    
    Defines the interface that all LLM models must implement to be used
    in the memory management system.
    """

    @abstractmethod
    async def generate(self, messages: Sequence[TextChatMessage], reasoning: bool = True) -> str:
        """
        Generate a response based on the provided messages.
        
        Args:
            messages: Sequence of chat messages to process
            reasoning: Whether to enable reasoning capabilities (default: True)
            
        Returns:
            Generated response as a string
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class MemoryRepository(ABC):
    """
    Abstract base class for memory storage implementations.
    
    Defines the interface for storing, retrieving, and managing memories
    in the memory management system.
    """

    @abstractmethod
    async def add_memory(self, memory: Memory) -> None:
        """
        Add a new memory to the repository.
        
        Args:
            memory: The memory to add
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    async def remove_memory(self, memory: Memory) -> None:
        """
        Remove a memory from the repository.
        
        Args:
            memory: The memory to remove
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    async def update_memory(self, memory: Memory) -> None:
        """
        Update an existing memory in the repository.
        
        Args:
            memory: The memory with updated content
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch_memory_by_name(self, name: str) -> Memory | None:
        """
        Retrieve a specific memory by its name.
        
        Args:
            name: The name of the memory to retrieve
            
        Returns:
            The memory with the specified name, or None if not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch_all_memories_abstract(self) -> Sequence[MemoryAbstract]:
        """
        Retrieve all memory abstracts from the repository.
        
        Returns lightweight MemoryAbstract objects containing only name and abstract,
        avoiding the potentially large memory_block content to reduce network and memory burden.
        
        Returns:
            Sequence of all stored memory abstracts
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class MemoryManager(ABC):
    """
    Abstract interface for memory management operations.
    
    Defines the interface for managing memories, including storage, retrieval,
    and intelligent updating of memories based on conversation context.
    
    The MemoryManager provides two types of operations:
    1. Basic operations: force_add_memory, force_update_memory, force_update_relevance_map
    2. High-level operations: Operations that combine basic operations with LLM capabilities
    """

    memory_repository: MemoryRepository
    visible_memories: Sequence[Memory]
    relevance_map: Mapping[str, int]

    @abstractmethod
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
        raise NotImplementedError
    @abstractmethod
    async def force_update_relevance_map(self, delta_map: Mapping[str, int]) -> "MemoryManager":
        """
        Update the relevance map with delta values.
        
        Args:
            delta_map: Mapping of memory names to delta relevance values to add
            
        Returns:
            New MemoryManager instance with updated relevance map
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def update_relevance_map(self, chat_messages: Sequence[TextChatMessage], delta: int) -> "MemoryManager":
        """
        Update relevance map based on memories related to chat messages.
        
        Args:
            chat_messages: Chat messages to analyze for related memories
            delta: The delta value to add to relevance counts
            
        Returns:
            New MemoryManager instance with updated relevance map
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
