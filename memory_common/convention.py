from abc import ABC, abstractmethod
from collections.abc import Sequence

from memory_common.model import Memory, MemoryAbstract, TextChatMessage


class LlmModel(ABC):
    @abstractmethod
    async def generate(self, messages: Sequence[TextChatMessage], reasoning: bool = True) -> str:
        raise NotImplementedError


class MemoryRepository(ABC):
    @abstractmethod
    async def add_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    @abstractmethod
    async def remove_memory_by_name(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def update_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fetch_memory_by_name(self, name: str) -> Memory | None:
        raise NotImplementedError

    @abstractmethod
    async def fetch_all_memories_abstract(self) -> Sequence[MemoryAbstract]:
        raise NotImplementedError


class MemorySession(ABC):
    @abstractmethod
    async def force_add_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    @abstractmethod
    async def force_update_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    @abstractmethod
    async def force_remove_memory_by_name(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def update_memory(self, chat_messages: Sequence[TextChatMessage]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def retrieve_context_memories(self) -> Sequence[Memory]:
        raise NotImplementedError

    @abstractmethod
    async def fetch_memory_by_name(self, name: str) -> Memory | None:
        raise NotImplementedError

    @abstractmethod
    async def fetch_all_memories_abstract(self) -> Sequence[MemoryAbstract]:
        raise NotImplementedError
