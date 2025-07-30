from collections.abc import MutableSequence, Sequence
from typing import Final, final, override

from memory_common.convention import MemoryRepository
from memory_common.model import Memory, MemoryAbstract


@final
class InMemoryMemoryRepository(MemoryRepository):
    """
    In-memory_common implementation of the MemoryRepository interface.
    
    Stores memories in a list for simple, non-persistent storage.
    Suitable for testing and development purposes.
    
    Attributes:
        _delegate: Internal list storing the memories
    """
    _delegate: Final[MutableSequence[Memory]]

    def __init__(self, delegate: Sequence[Memory] | None = None):
        """
        Initialize the in-memory_common repository.
        
        Args:
            delegate: Optional initial sequence of memories to store
        """
        if delegate is None:
            delegate = []
        self._delegate = []
        for memory in delegate:
            self.__add_memory_impl(memory)

    def __add_memory_impl(self, memory: Memory) -> None:
        """
        Internal method to add a memory_common with duplicate checking.
        
        Args:
            memory: The memory_common to add
            
        Raises:
            ValueError: If a memory_common with the same name already exists
        """
        if memory in self._delegate:
            raise ValueError(f"Memory with name {memory.name} already exists")
        self._delegate.append(memory)

    @override
    async def add_memory(self, memory: Memory) -> None:
        """
        Add a new memory_common to the repository.
        
        Args:
            memory: The memory_common to add
            
        Raises:
            ValueError: If a memory_common with the same name already exists
        """
        self.__add_memory_impl(memory)

    @override
    async def remove_memory_by_name(self, name: str) -> None:
        """
        Remove a memory_common from the repository.
        
        Args:
            name: The memory_common to remove
            
        Raises:
            ValueError: If the memory_common is not found in the repository
        """
        self._delegate.remove([memory for memory in self._delegate if memory.name == name][0])

    @override
    async def update_memory(self, memory: Memory) -> None:
        """
        Update an existing memory_common in the repository.
        
        Removes the old version and adds the updated version.
        
        Args:
            memory: The memory_common with updated content
            
        Raises:
            ValueError: If the memory_common is not found or if duplicate names exist
        """
        self._delegate.remove(memory)
        self.__add_memory_impl(memory)

    @override
    async def fetch_memory_by_name(self, name: str) -> Memory | None:
        """
        Retrieve a specific memory_common by its name.
        
        Args:
            name: The name of the memory_common to retrieve
            
        Returns:
            The memory_common with the specified name, or None if not found
        """
        for memory in self._delegate:
            if memory.name == name:
                return memory
        return None

    @override
    async def fetch_all_memories_abstract(self) -> Sequence[MemoryAbstract]:
        """
        Retrieve all memory_common abstracts from the repository.
        
        Returns lightweight MemoryAbstract objects containing only name and abstract,
        avoiding the potentially large memory_block content to reduce network and memory_common burden.
        
        Returns:
            Sequence of all stored memory_common abstracts
        """
        return [MemoryAbstract(name=memory.name, abstract=memory.abstract) for memory in self._delegate]
