"""
Memory Management System Demo

This module demonstrates the core functionality of the memory management system,
showing how memories can be updated based on chat conversations using LLM analysis.

The demo creates initial memories, simulates a chat conversation, and then
updates the memories based on new information from the conversation.
"""

from app.llm_model import QwenModel
from memory.in_memory import InMemoryMemoryRepository
from memory.llm_ability import LlmAbility
from memory.manager import MemoryManager
from memory.model import Memory, TextChatMessage


async def update_memory() -> None:
    """
    Demonstrate the memory update functionality.
    
    Creates initial memories, simulates a chat conversation with new information,
    and shows how the system can intelligently update memories based on the
    conversation context using LLM analysis.
    """
    print("=== 记忆更新能力测试Demo ===\n")

    # 初始化组件
    llm_model = QwenModel()
    memory_repository = InMemoryMemoryRepository()
    llm_ability = LlmAbility(llm_model)

    # 创建测试数据 - 初始记忆
    initial_memories = [
        Memory(
            name="user_preferences",
            abstract="用户的个人偏好和兴趣",
            memory_block="用户喜欢阅读科幻小说，特别是刘慈欣的作品。对编程和人工智能很感兴趣。"
        ),
        Memory(
            name="conversation_context",
            abstract="对话的上下文和背景信息",
            memory_block="这是我们第一次对话，用户刚开始了解这个记忆系统。"
        ),
        Memory(
            name="technical_knowledge",
            abstract="用户的技术知识水平",
            memory_block="用户具备基础的编程知识，了解Python语言。"
        )
    ]

    # 添加初始记忆到存储库
    for memory in initial_memories:
        await memory_repository.add_memory(memory)

    # 创建聊天历史 - 模拟用户与AI的对话，包含新信息
    chat_history = [
        TextChatMessage(role="user", text="你好，我是一名机器学习工程师"),
        TextChatMessage(role="assistant", text="你好！很高兴认识你。作为机器学习工程师，你一定对AI技术有深入的了解。"),
        TextChatMessage(role="user", text="是的，我最近在研究大语言模型，特别是Transformer架构"),
        TextChatMessage(role="assistant", text="Transformer确实是现代NLP的核心架构。你在这方面有什么具体的项目经验吗？"), # noqa: E501
        TextChatMessage(role="user", text="我正在开发一个智能对话系统，希望它能记住我们之前的对话内容"),
        TextChatMessage(role="assistant", text="这正是记忆系统要解决的问题！你提到的需求很有意思。")
    ]

    # 创建记忆作用域
    memory_scope = MemoryManager(
        memory_repository=memory_repository,
        visible_chat_messages=chat_history,
        visible_memories=initial_memories,
        llm_ability=llm_ability
    )

    print("📝 初始记忆状态:")
    for i, memory in enumerate(memory_scope.visible_memories, 1):
        print(f"{i}. 记忆名称: {memory.name}")
        print(f"   摘要: {memory.abstract}")
        print(f"   内容: {memory.memory_block}")
        print()

    print("💬 聊天历史:")
    for i, message in enumerate(chat_history, 1):
        print(f"{i}. {message.role}: {message.text}")
    print()

    print("🔄 开始更新记忆...")
    print("正在分析聊天历史，确定需要更新的记忆...")

    try:
        # 执行记忆更新
        updated_memory_scope = await memory_scope.update_all_memories()

        print("✅ 记忆更新完成！\n")

        print("📝 更新后的记忆状态:")
        for i, memory in enumerate(updated_memory_scope.visible_memories, 1):
            print(f"{i}. 记忆名称: {memory.name}")
            print(f"   摘要: {memory.abstract}")
            print(f"   内容: {memory.memory_block}")
            print()

        print("🔍 记忆变化对比:")
        for old_memory in memory_scope.visible_memories:
            for new_memory in updated_memory_scope.visible_memories:
                if old_memory.name == new_memory.name:
                    if old_memory.memory_block != new_memory.memory_block:
                        print(f"📋 '{old_memory.name}' 记忆已更新:")
                        print(f"   旧内容: {old_memory.memory_block}")
                        print(f"   新内容: {new_memory.memory_block}")
                        print()
                    else:
                        print(f"📋 '{old_memory.name}' 记忆无变化")
                        print()

    except Exception as e:
        print(f"❌ 记忆更新过程中出现错误: {e}")
        print("这可能是由于API配置问题或网络连接问题导致的。")
        print("请检查.env文件中的DASHSCOPE_API_KEY配置。")


async def new_memory() -> None:
    """
    Demonstrate the new memory creation functionality.
    
    Creates initial memories representing a mid-conversation state, simulates 
    recent chat messages containing new types of information not covered by 
    existing memories, and shows how the system can intelligently create new 
    memory blocks based on the conversation context using LLM analysis.
    """
    print("=== 新记忆创建能力测试Demo ===\n")

    # 初始化组件
    llm_model = QwenModel()
    memory_repository = InMemoryMemoryRepository()
    llm_ability = LlmAbility(llm_model)

    # 创建测试数据 - 初始记忆（模拟对话进行到一半的状态）
    initial_memories = [
        Memory(
            name="user_background",
            abstract="用户的基本背景信息",
            memory_block="用户是一名软件工程师，有5年的开发经验，主要使用Java和Python。"
        ),
        Memory(
            name="work_projects",
            abstract="用户当前的工作项目",
            memory_block="用户正在开发一个电商平台的后端系统，使用Spring Boot框架。"
        ),
        Memory(
            name="learning_interests",
            abstract="用户的学习兴趣和目标",
            memory_block="用户对机器学习很感兴趣，希望将AI技术应用到自己的项目中。"
        )
    ]

    # 添加初始记忆到存储库
    for memory in initial_memories:
        await memory_repository.add_memory(memory)

    # 创建聊天历史 - 只保留最近几条消息，包含新类型的信息
    # 这些消息包含了现有记忆中没有涵盖的新信息类型
    chat_history = [
        TextChatMessage(role="user", text="对了，我忘了告诉你，我其实还有一个副业"),
        TextChatMessage(role="assistant", text="哦？什么样的副业呢？听起来很有意思。"),
        TextChatMessage(role="user", text="我在业余时间做摄影师，主要拍摄婚礼和活动"),
        TextChatMessage(role="assistant", text="摄影师！这是一个很有创意的副业。你做这个多久了？"),
        TextChatMessage(role="user", text="大概3年了，现在每个月能有额外的收入，而且我特别喜欢捕捉人们的美好瞬间"),
        TextChatMessage(role="assistant", text="这真是一个很棒的爱好兼职业。摄影确实能让人发现生活中的美。")
    ]

    # 创建记忆作用域
    memory_scope = MemoryManager(
        memory_repository=memory_repository,
        visible_chat_messages=chat_history,
        visible_memories=initial_memories,
        llm_ability=llm_ability
    )

    print("📝 初始记忆状态:")
    for i, memory in enumerate(memory_scope.visible_memories, 1):
        print(f"{i}. 记忆名称: {memory.name}")
        print(f"   摘要: {memory.abstract}")
        print(f"   内容: {memory.memory_block}")
        print()

    print("💬 最近的聊天历史:")
    for i, message in enumerate(chat_history, 1):
        print(f"{i}. {message.role}: {message.text}")
    print()

    print("🆕 开始创建新记忆...")
    print("正在分析聊天历史，识别现有记忆未涵盖的新信息...")

    try:
        # 执行新记忆创建
        updated_memory_scope = await memory_scope.create_new_memories()

        print("✅ 新记忆创建完成！\n")

        print("📝 更新后的记忆状态:")
        for i, memory in enumerate(updated_memory_scope.visible_memories, 1):
            print(f"{i}. 记忆名称: {memory.name}")
            print(f"   摘要: {memory.abstract}")
            print(f"   内容: {memory.memory_block}")
            print()

        # 显示新创建的记忆
        original_memory_names = {memory.name for memory in memory_scope.visible_memories}
        new_memories = [
            memory for memory in updated_memory_scope.visible_memories 
            if memory.name not in original_memory_names
        ]

        if new_memories:
            print("🎉 新创建的记忆:")
            for i, memory in enumerate(new_memories, 1):
                print(f"{i}. 记忆名称: {memory.name}")
                print(f"   摘要: {memory.abstract}")
                print(f"   内容: {memory.memory_block}")
                print()
        else:
            print("📋 没有创建新的记忆，现有记忆已经涵盖了所有信息。")

        print("📊 记忆统计:")
        print(f"   原始记忆数量: {len(memory_scope.visible_memories)}")
        print(f"   更新后记忆数量: {len(updated_memory_scope.visible_memories)}")
        print(f"   新增记忆数量: {len(new_memories)}")

    except Exception as e:
        print(f"❌ 新记忆创建过程中出现错误: {e}")
        print("这可能是由于API配置问题或网络连接问题导致的。")
        print("请检查.env文件中的DASHSCOPE_API_KEY配置。")
