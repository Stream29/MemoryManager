from asyncio import gather
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import Final, final

from memory.convention import MemoryManager, MemoryRepository
from memory.llm_ability import LlmAbility
from memory.model import Memory, MemoryAbstract, TextChatMessage


@final
class MemoryManagerImpl(MemoryManager):
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
        self.memory_repository = memory_repository
        self.visible_memories = visible_memories
        self._llm_ability = llm_ability
        self.relevance_map = relevance_map or {}

    async def force_add_memory(self, memory: Memory) -> MemoryManager:
        # Check if memory with same name already exists
        if any(x.name == memory.name for x in self.visible_memories):
            raise ValueError(f"Memory with name {memory.name} already exists")

        await self.memory_repository.add_memory(memory)
        return MemoryManagerImpl(
            memory_repository=self.memory_repository,
            visible_memories=[*self.visible_memories, memory],
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def force_update_relevance_map(self, delta_map: Mapping[str, int]) -> MemoryManager:
        new_relevance_map: Final[MutableMapping[str, int]] = {**self.relevance_map, **delta_map}
        for name, delta_value in delta_map.items():
            if name in new_relevance_map:
                new_relevance_map[name] += delta_value
            else:
                new_relevance_map[name] = delta_value
        return MemoryManagerImpl(
            memory_repository=self.memory_repository,
            visible_memories=self.visible_memories,
            llm_ability=self._llm_ability,
            relevance_map=new_relevance_map
        )

    async def force_update_memory(self, memory: Memory) -> MemoryManager:
        new_visible_memories: MutableSequence[Memory] = [*self.visible_memories]
        new_visible_memories.remove(memory)
        new_visible_memories.append(memory)
        await self.memory_repository.update_memory(memory)
        return MemoryManagerImpl(
            memory_repository=self.memory_repository,
            visible_memories=new_visible_memories,
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def full_update(
            self,
            chat_messages: Sequence[TextChatMessage],
            delta: int
    ) -> MemoryManager:
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
        new_memory_manager: MemoryManager = self
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
    ) -> MemoryManager:
        updated_memories: Final[Sequence[Memory]] = await self._get_updated_memories(
            current_memories=await self.memory_repository.fetch_all_memories_abstract(),
            chat_messages=chat_messages,
        )

        # Apply all updates to create new memory manager
        new_memory_manager: MemoryManager = self
        for memory in updated_memories:
            new_memory_manager = await new_memory_manager.force_update_memory(memory)
        return new_memory_manager

    async def extract_new_memories(
            self,
            chat_messages: Sequence[TextChatMessage]
    ) -> MemoryManager:
        new_memories = await self._llm_ability.extract_new_memories(
            await self.memory_repository.fetch_all_memories_abstract(),
            chat_messages
        )

        # Add all new memories to the manager
        new_manager: MemoryManager = self
        for memory in new_memories:
            new_manager = await new_manager.force_add_memory(memory)

        return new_manager

    async def refresh_visible_memory_list(self, limit: int) -> MemoryManager:
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

        return MemoryManagerImpl(
            memory_repository=self.memory_repository,
            visible_memories=visible_memories,
            llm_ability=self._llm_ability,
            relevance_map=self.relevance_map
        )

    async def update_relevance_map(self, chat_messages: Sequence[TextChatMessage], delta: int) -> MemoryManager:
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
    ) -> MemoryManager:
        updated_manager = await self.update_relevance_map(chat_messages, delta)
        return await updated_manager.refresh_visible_memory_list(limit)

    async def _get_updated_memory(
            self,
            memory_abstract: MemoryAbstract,
            chat_messages: Sequence[TextChatMessage]
    ) -> Memory:
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
