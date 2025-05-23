# langchain_agent_demo.py

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os

# 加载环境变量中的 OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 初始化语言模型（ChatGPT）
llm = OpenAI(temperature=0, openai_api_key=openai_key)

# 初始化搜索引擎（DuckDuckGo）
search = DuckDuckGoSearchAPIWrapper()

# 构建工具列表（供智能 Agent 使用）
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="适合用来检索某个技术或新闻主题的相关网页内容"
    )
]

# 构建智能 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 运行测试：自动检索并总结 LangChain 主题
question = "LangChain 是什么？它适合用来做什么？请你检索资料并总结核心用途"
response = agent.run(question)

# 输出结果
print("\n>>> 最终总结结果：\n")
print(response)
