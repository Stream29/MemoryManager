from collections.abc import Sequence
from typing import Final, final, override

from pydantic import BaseModel


@final
class Memory(BaseModel):
    """
    Represents a memory_common unit in the memory_common management system.
    
    A memory_common consists of a unique name, an abstract summary, and the actual memory_common content.
    Memories are considered equal if they have the same name.
    
    Attributes:
        name: Unique identifier for the memory_common
        abstract: Brief summary or description of the memory_common content
        memory_block: The actual memory_common content/data
    """
    name: Final[str] # type: ignore[misc]
    abstract: Final[str] # type: ignore[misc]
    memory_block: Final[str] # type: ignore[misc]
    
    @override
    def __eq__(self, other: object) -> bool:
        """
        Compare memories based on their names.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if both objects are Memory instances with the same name
        """
        if isinstance(other, Memory):
            return self.name == other.name
        return False

@final
class MemoryAbstract(BaseModel):
    """
    Represents a lightweight version of a memory_common containing only name and abstract.
    
    Used when full memory_common content is not needed, such as in requests to determine
    which memories need updating.
    
    Attributes:
        name: Unique identifier for the memory_common
        abstract: Brief summary or description of the memory_common content
    """
    name: Final[str] # type: ignore[misc]
    abstract: Final[str] # type: ignore[misc]

@final
class TextChatMessage(BaseModel):
    """
    Represents a single message in a chat conversation.
    
    Used to store chat history for memory_common update operations.
    
    Attributes:
        role: The role of the message sender (e.g., "user", "assistant", "system")
        text: The actual message content
    """
    role: Final[str] # type: ignore[misc]
    text: Final[str] # type: ignore[misc]

@final
class UpdateMemoriesRequest(BaseModel):
    """
    Request model for determining which memories need to be updated.
    
    Contains chat history and existing memory_common abstracts to help the LLM
    decide which memories should be updated based on new information.
    
    Attributes:
        chat_history: Sequence of chat messages providing context
        old_memory: Existing memory_common abstracts to evaluate for updates
    """
    chat_history: Final[Sequence[TextChatMessage]] # type: ignore[misc]
    old_memory: Final[Sequence[MemoryAbstract]] # type: ignore[misc]

@final
class UpdateMemoriesResponse(BaseModel):
    """
    Response model containing names of memories that need updating.
    
    Attributes:
        memories_to_update: Names of memories that should be updated
    """
    memories_to_update: Final[Sequence[str]] # type: ignore[misc]

@final
class UpdateSingleMemoryRequest(BaseModel):
    """
    Request model for updating a specific memory_common.
    
    Contains chat history and the existing memory_common to generate an updated version.
    
    Attributes:
        chat_history: Sequence of chat messages providing new context
        old_memory: The existing memory_common to be updated
    """
    chat_history: Final[Sequence[TextChatMessage]] # type: ignore[misc]
    old_memory: Final[Memory] # type: ignore[misc]

@final
class UpdateSingleMemoryResponse(BaseModel):
    """
    Response model containing the updated memory_common content.
    
    Attributes:
        new_memory_block: The updated memory_common content
    """
    new_memory_block: Final[str] # type: ignore[misc]

@final
class CreateNewMemoriesRequest(BaseModel):
    """
    Request model for creating new memories based on chat history.
    
    Contains current existing memories and chat history to help the LLM
    determine what new memories should be created for uncovered information.
    
    Attributes:
        current_memories: Existing memory_common abstracts to check against
        chat_history: Chat messages that may contain new information
    """
    current_memories: Final[Sequence[MemoryAbstract]] # type: ignore[misc]
    chat_history: Final[Sequence[TextChatMessage]] # type: ignore[misc]

@final
class CreateNewMemoriesResponse(BaseModel):
    """
    Response model containing new memories to be created.
    
    Attributes:
        new_memories: New memory_common objects that should be created
    """
    new_memories: Final[Sequence[Memory]] # type: ignore[misc]

@final
class FindAssociatedMemoriesRequest(BaseModel):
    """
    Request model for finding memories associated with chat messages.
    
    Contains current existing memories and chat messages to help the LLM
    determine which memories are most relevant to the current conversation.
    
    Attributes:
        current_memories: Existing memory_common abstracts to evaluate for relevance
        chat_messages: Chat messages to find associations with
    """
    current_memories: Final[Sequence[MemoryAbstract]] # type: ignore[misc]
    chat_messages: Final[Sequence[TextChatMessage]] # type: ignore[misc]

@final
class FindAssociatedMemoriesResponse(BaseModel):
    """
    Response model containing names of memories associated with the conversation.
    
    Attributes:
        associated_memories: Names of memories that are associated with the chat messages
    """
    associated_memories: Final[Sequence[str]] # type: ignore[misc]
