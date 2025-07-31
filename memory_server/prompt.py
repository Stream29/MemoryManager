from typing import Final

update_memories_system_prompt: Final[str] = """
Both your input and output should be in JSON format.

! Below is the schema for input content !
{
    "type": "object",
    "properties": {
        "chat_history": {
            "type": "array",
            "description": "Chat history of the user and LLM assistant.",
            "items": {
                "type": "object",
                "description": "A message between the chat of the user and the LLM.",
                "properties": {
                    "role": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    }
                },
                "required": [
                    "role",
                    "text"
                ]
            }
        },
        "old_memory": {
            "type": "array",
            "description": "The current memory_common about the conversation.",
            "items": {
                "type": "object",
                "description": "A memory_common block that stores information about the conversation on specific topics.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The unique name of the memory_common block, used to identify it."
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Tells what the memory_common block is about."
                    }
                },
                "required": [
                    "name",
                    "abstract"
                ]
            }
        }
    },
    "required": [
        "chat_history",
        "old_memory"
    ]
}
! Above is the schema for input content !

! Below is the schema for output content !
{
    "type": "object",
    "description": "You should decide which part of the memory_common to update.\nOnly the memory_common listed in old_memory is considered.\nYou shouldn't create new memory_common, just update the existing ones.\nYou only need to list the `name` of the memory_common.\n",
    "properties": {
        "memories_to_update": {
            "type": "array",
            "description": "List of names of memory_common to update. if no memory_common need to be updated, leave it empty list as []",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "memories_to_update"
    ]
}
! Above is the schema for output content !

Your output must strictly follow the schema format, do not output any content outside of the JSON body.
""" # noqa: E501

update_single_memory_system_prompt: Final[str] = """
Both your input and output should be in JSON format.

! Below is the schema for input content !
{
    "type": "object",
    "description": "You need to update the content of memory_common block according to `chat_history`",
    "properties": {
        "chat_history": {
            "type": "array",
            "description": "Chat history of the user and LLM assistant.",
            "items": {
                "type": "object",
                "description": "A message between the chat of the user and the LLM.",
                "properties": {
                    "role": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    }
                },
                "required": [
                    "role",
                    "text"
                ]
            }
        },
        "old_memory": {
            "type": "object",
            "description": "A memory_common block that stores information about the conversation on specific topics.",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The unique name of the memory_common block, used to identify it."
                },
                "abstract": {
                    "type": "string",
                    "description": "Tells what the memory_common block is about."
                },
                "memory_block": {
                    "type": "string",
                    "description": "The content of the memory_common block."
                }
            },
            "required": [
                "name",
                "abstract",
                "memory_block"
            ]
        }
    },
    "required": [
        "chat_history",
        "old_memory"
    ]
}
! Above is the schema for input content !

! Below is the schema for output content !
{
    "type": "object",
    "properties": {
        "new_memory_block": {
            "type": "string",
            "description": "The updated memory_common block. You should keep all the information in the old memory_common block, and new information from `chat_history`. You should address user as 'user' and LLM as 'assistant'."
        }
    },
    "required": [
        "new_memory_block"
    ]
}
! Above is the schema for output content !

Your output must strictly follow the schema format, do not output any content outside of the JSON body.
""" # noqa: E501

new_memory_system_prompt: Final[str] = """
Both your input and output should be in JSON format.

! Below is the schema for input content !
{
    "type": "object",
    "properties": {
        "current_memories": {
            "type": "array",
            "description": "The memory_common about the conversation that needs to be checked.",
            "items": {
                "type": "object",
                "description": "A memory_common block that stores information about the conversation on specific topics.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The unique name of the memory_common block, used to identify it."
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Tells what the memory_common block is about."
                    }
                },
                "required": [
                    "name",
                    "abstract"
                ]
            }
        },
        "chat_history": {
            "type": "array",
            "description": "Chat history of the user and LLM assistant that may contain new information.",
            "items": {
                "type": "object",
                "description": "A message between the chat of the user and the LLM.",
                "properties": {
                    "role": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    }
                },
                "required": [
                    "role",
                    "text"
                ]
            }
        }
    },
    "required": [
        "current_memories",
        "chat_history"
    ]
}
! Above is the schema for input content !

! Below is the schema for output content !
{
    "type": "object",
    "description": "You should declare clear boundaries for the new memory_common blocks in `abstract` and give them a clear and unique name.",
    "properties": {
        "new_memories": {
            "type": "array",
            "description": "The new memories that contains new information that not covered by the old memory_common blocks. If all information is covered by current memory_common blocks, leave it empty list as [].",
            "items": {
                "type": "object",
                "description": "A memory_common block that stores information about the conversation on specific topics.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The unique name of the memory_common block, used to identify it."
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Tells what the memory_common block is about."
                    },
                    "memory_block": {
                        "type": "string",
                        "description": "The content of the memory_common block. You should address user as 'user' and LLM as 'assistant'."
                    }
                },
                "required": [
                    "name",
                    "abstract",
                    "memory_block"
                ]
            }
        }
    },
    "required": [
        "new_memories"
    ]
}
! Above is the schema for output content !
""" # noqa: E501
