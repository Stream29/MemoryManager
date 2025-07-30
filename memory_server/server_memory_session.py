from asyncio import gather
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Final, final

from memory_common.convention import MemoryRepository, MemorySession
from memory_common.model import Memory, MemoryAbstract, TextChatMessage
from memory_server.llm_ability import LlmAbility


@final
class ServerMemorySession(MemorySession):
    _memory_repository: Final[MemoryRepository]
    _llm_ability: Final[LlmAbility]
    _relevance_map: Final[MutableMapping[str, int]]

    def __init__(
            self,
            memory_repository: MemoryRepository,
            llm_ability: LlmAbility,
            relevance_map: Mapping[str, int] | None = None
    ):
        self._memory_repository = memory_repository
        self._llm_ability = llm_ability
        self._relevance_map = {**relevance_map} if relevance_map else {}

    async def force_remove_memory_by_name(self, name: str) -> None:
        await self._memory_repository.remove_memory_by_name(name)

    async def fetch_memory_by_name(self, name: str) -> Memory | None:
        return await self._memory_repository.fetch_memory_by_name(name)

    async def fetch_all_memories_abstract(self) -> Sequence[MemoryAbstract]:
        return await self._memory_repository.fetch_all_memories_abstract()

    async def force_add_memory(self, memory: Memory) -> None:
        await self._memory_repository.add_memory(memory)

    async def force_update_memory(self, memory: Memory) -> None:
        await self._memory_repository.update_memory(memory)

    async def retrieve_context_memories(self) -> Sequence[Memory]:
        result = [await self._memory_repository.fetch_memory_by_name(memory.name)
                  for memory in await self._memory_repository.fetch_all_memories_abstract()]
        return [memory for memory in result if memory is not None]

    async def update_memory(
            self,
            chat_messages: Sequence[TextChatMessage]
    ) -> None:
        current_memory_list: Final[
            Sequence[MemoryAbstract]] = await self._memory_repository.fetch_all_memories_abstract()
        new_memories, updated_memories = await gather(
            self._llm_ability.extract_new_memories(
                current_memories=current_memory_list,
                chat_messages=chat_messages
            ),
            self._get_updated_memories(current_memory_list, chat_messages)
        )
        for new_memory in new_memories:
            await self.force_add_memory(new_memory)
        for updated_memory in updated_memories:
            await self.force_update_memory(updated_memory)

    async def _get_updated_memory(
            self,
            memory_abstract: MemoryAbstract,
            chat_messages: Sequence[TextChatMessage]
    ) -> Memory:
        old_memory: Final[Memory | None] = await self._memory_repository.fetch_memory_by_name(memory_abstract.name)
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
