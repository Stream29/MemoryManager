from asyncio import run

from app.llm_model import generate
from memory.model import TextChatMessage


async def main() -> None:
    print(await generate([TextChatMessage("user", "背诵岳阳楼记")]))



if __name__ == "__main__":
    run(main())
