from abc import ABC, abstractmethod
from collections.abc import Sequence

from memory.model import Memory, TextChatMessage


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
    async def fetch_all_memories(self) -> Sequence[Memory]:
        """
        Retrieve all memories from the repository.
        
        Returns:
            Sequence of all stored memories
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError