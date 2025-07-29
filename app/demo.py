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
        updated_memory_scope = await memory_scope.update_existing_memories(chat_history)

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
        updated_memory_scope = await memory_scope.extract_new_memories(chat_history)

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


async def memory_relevance_sorting() -> None:
    """
    Demonstrate the memory relevance sorting functionality.
    
    Creates initial memories, simulates three different conversation scenarios,
    and shows how the system intelligently sorts and selects the most relevant
    memories based on conversation context using LLM analysis and relevance counting.
    """
    print("=== 记忆相关度排序更新测试Demo ===\n")

    # 初始化组件
    llm_model = QwenModel()
    memory_repository = InMemoryMemoryRepository()
    llm_ability = LlmAbility(llm_model)

    # 创建测试数据 - 初始记忆（创建更多记忆以展示排序效果）
    initial_memories = [
        Memory(
            name="user_work_background",
            abstract="用户的工作背景和职业信息",
            memory_block="用户是一名软件工程师，有3年的开发经验，主要使用Java和Python进行后端开发。"
        ),
        Memory(
            name="user_hobbies",
            abstract="用户的兴趣爱好",
            memory_block="用户喜欢阅读科幻小说，特别喜欢《三体》系列。业余时间喜欢打篮球和听音乐。"
        ),
        Memory(
            name="user_learning_goals",
            abstract="用户的学习目标和计划",
            memory_block="用户希望学习机器学习和人工智能技术，计划在未来一年内转向AI领域。"
        ),
        Memory(
            name="user_travel_experience",
            abstract="用户的旅行经历",
            memory_block="用户去年去了日本旅行，对日本的文化和美食印象深刻，特别喜欢京都的古建筑。"
        ),
        Memory(
            name="user_food_preferences",
            abstract="用户的饮食偏好",
            memory_block="用户喜欢吃辣的食物，特别是川菜和湘菜。不太喜欢甜食，但对咖啡很有研究。"
        ),
        Memory(
            name="user_tech_interests",
            abstract="用户的技术兴趣和关注点",
            memory_block="用户对云计算和微服务架构很感兴趣，最近在学习Docker和Kubernetes。"
        )
    ]

    # 添加初始记忆到存储库
    for memory in initial_memories:
        await memory_repository.add_memory(memory)

    # 创建初始记忆作用域
    memory_scope = MemoryManager(
        memory_repository=memory_repository,
        visible_memories=initial_memories,
        llm_ability=llm_ability
    )

    print("📝 初始记忆状态:")
    for i, memory in enumerate(memory_scope.visible_memories, 1):
        print(f"{i}. 记忆名称: {memory.name}")
        print(f"   摘要: {memory.abstract}")
        print(f"   内容: {memory.memory_block}")
        print()

    print(f"📊 初始相关度计数: {memory_scope.relevance_map}")
    print()

    # 第一次对话更新 - 关于工作和技术的对话
    print("🔄 第一次对话更新 - 工作和技术话题")
    print("=" * 50)
    
    first_chat = [
        TextChatMessage(role="user", text="最近在公司里开始使用Docker容器化部署"),
        TextChatMessage(role="assistant", text="Docker确实是现代开发中很重要的技术。你们是怎么应用的？"),
        TextChatMessage(role="user", text="我们把Java应用打包成Docker镜像，然后部署到Kubernetes集群上"),
        TextChatMessage(role="assistant", text="这是很好的实践！Kubernetes能很好地管理容器化应用。")
    ]

    print("💬 第一次对话内容:")
    for i, message in enumerate(first_chat, 1):
        print(f"{i}. {message.role}: {message.text}")
    print()

    try:
        # 执行第一次记忆相关度更新，保留前3个最相关的记忆
        updated_scope_1 = await memory_scope.update_visible_memory_list(first_chat, limit=3)
        
        print("✅ 第一次更新完成！")
        print(f"📊 更新后相关度计数: {updated_scope_1.relevance_map}")
        print()
        
        print("📝 第一次更新后可见记忆 (前3个最相关):")
        for i, memory in enumerate(updated_scope_1.visible_memories, 1):
            relevance_count = updated_scope_1.relevance_map.get(memory.name, 0)
            print(f"{i}. 记忆名称: {memory.name} (相关度: {relevance_count})")
            print(f"   摘要: {memory.abstract}")
            print()

        # 第二次对话更新 - 关于旅行和文化的对话
        print("🔄 第二次对话更新 - 旅行和文化话题")
        print("=" * 50)
        
        second_chat = [
            TextChatMessage(role="user", text="我在考虑下次假期去哪里旅行"),
            TextChatMessage(role="assistant", text="有什么特别想去的地方吗？"),
            TextChatMessage(role="user", text="想去欧洲看看，特别是意大利的古建筑和艺术"),
            TextChatMessage(role="assistant", text="意大利确实有很多历史悠久的建筑，就像你之前去日本时看到的京都古建筑一样。") # noqa: E501
        ]

        print("💬 第二次对话内容:")
        for i, message in enumerate(second_chat, 1):
            print(f"{i}. {message.role}: {message.text}")
        print()

        # 执行第二次记忆相关度更新
        updated_scope_2 = await updated_scope_1.update_visible_memory_list(second_chat, limit=3)
        
        print("✅ 第二次更新完成！")
        print(f"📊 更新后相关度计数: {updated_scope_2.relevance_map}")
        print()
        
        print("📝 第二次更新后可见记忆 (前3个最相关):")
        for i, memory in enumerate(updated_scope_2.visible_memories, 1):
            relevance_count = updated_scope_2.relevance_map.get(memory.name, 0)
            print(f"{i}. 记忆名称: {memory.name} (相关度: {relevance_count})")
            print(f"   摘要: {memory.abstract}")
            print()

        # 第三次对话更新 - 关于学习和AI的对话
        print("🔄 第三次对话更新 - 学习和AI话题")
        print("=" * 50)
        
        third_chat = [
            TextChatMessage(role="user", text="我最近开始学习机器学习了"),
            TextChatMessage(role="assistant", text="太好了！这正符合你的学习目标。你从哪个方面开始学的？"),
            TextChatMessage(role="user", text="先从Python的机器学习库开始，比如scikit-learn"),
            TextChatMessage(role="assistant", text="这是很好的起点！结合你的Python开发经验，应该会学得很快。")
        ]

        print("💬 第三次对话内容:")
        for i, message in enumerate(third_chat, 1):
            print(f"{i}. {message.role}: {message.text}")
        print()

        # 执行第三次记忆相关度更新
        updated_scope_3 = await updated_scope_2.update_visible_memory_list(third_chat, limit=3)
        
        print("✅ 第三次更新完成！")
        print(f"📊 更新后相关度计数: {updated_scope_3.relevance_map}")
        print()
        
        print("📝 第三次更新后可见记忆 (前3个最相关):")
        for i, memory in enumerate(updated_scope_3.visible_memories, 1):
            relevance_count = updated_scope_3.relevance_map.get(memory.name, 0)
            print(f"{i}. 记忆名称: {memory.name} (相关度: {relevance_count})")
            print(f"   摘要: {memory.abstract}")
            print()

        # 显示三次更新的效果对比
        print("📈 三次更新效果对比:")
        print("=" * 50)
        
        print("🔍 相关度变化趋势:")
        all_memory_names: set[str] = set()
        for scope in [memory_scope, updated_scope_1, updated_scope_2, updated_scope_3]:
            all_memory_names.update(scope.relevance_map.keys())
        
        for memory_name in sorted(all_memory_names):
            counts = [
                memory_scope.relevance_map.get(memory_name, 0),
                updated_scope_1.relevance_map.get(memory_name, 0),
                updated_scope_2.relevance_map.get(memory_name, 0),
                updated_scope_3.relevance_map.get(memory_name, 0)
            ]
            print(f"📋 {memory_name}: {counts[0]} → {counts[1]} → {counts[2]} → {counts[3]}")
        
        print()
        print("🎯 可见记忆变化:")
        scopes = [
            ("初始状态", memory_scope),
            ("第一次更新后", updated_scope_1),
            ("第二次更新后", updated_scope_2),
            ("第三次更新后", updated_scope_3)
        ]
        
        for stage_name, scope in scopes:
            visible_names = [memory.name for memory in scope.visible_memories]
            print(f"📌 {stage_name}: {visible_names}")

    except Exception as e:
        print(f"❌ 记忆相关度排序过程中出现错误: {e}")
        print("这可能是由于API配置问题或网络连接问题导致的。")
        print("请检查.env文件中的DASHSCOPE_API_KEY配置。")
