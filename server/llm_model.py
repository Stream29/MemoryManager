import asyncio
import os
from asyncio import Queue, create_task
from collections.abc import Sequence
from typing import Final, Literal, cast, final, override

from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from memory_common.convention import LlmModel
from memory_common.model import TextChatMessage

load_dotenv()

lock: Final[asyncio.Lock] = asyncio.Lock()

openai_client: Final[AsyncOpenAI] = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

async def generate(messages: Sequence[TextChatMessage], reasoning: bool = True, ) -> str:
    result: Final[ChatCompletion | AsyncStream[ChatCompletionChunk]] = await openai_client.chat.completions.create(
        model=cast(Literal["gpt-4o"], "qwen-turbo-2025-07-15"),
        messages=[
            cast(ChatCompletionMessageParam, {"role": message.role, "content": message.text}) for message in messages
        ],
        stream=True,
        extra_body={"enable_thinking": str(reasoning)}
    )
    if isinstance(result, ChatCompletion):
        raise ValueError("Expected a stream response")
    buffer: str = ""
    queue: Queue[str | None] = Queue()

    async def print_worker() -> None:
        async with lock:
            while True:
                text_chunk: str | None = await queue.get()
                if text_chunk is None:
                    break
                print(text_chunk, end="", flush=True)
            print()

    task = create_task(print_worker())

    async for chunk in result:
        delta: ChoiceDelta = chunk.choices[0].delta
        text: str = delta.content or ""
        buffer += text
        queue.put_nowait(f"\x1b[2m{text}\x1b[0m")
        reasoning_content: str = (delta.model_extra or {}).get("reasoning_content") or ""
        queue.put_nowait(f"\x1b[33m{reasoning_content}\x1b[0m")
    queue.put_nowait(None)
    await task
    await asyncio.sleep(1.0)
    return buffer


@final
class QwenModel(LlmModel):
    @override
    async def generate(self, messages: Sequence[TextChatMessage], reasoning: bool = True) -> str:
        return await generate(messages, reasoning=reasoning)
