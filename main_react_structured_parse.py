# uv add python-dotenv black isort langchain langchain-core langchain-ollama langchainhub langchain-community
from dotenv import load_dotenv

load_dotenv()

import os

# The agents are now in langchain_classic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langsmith import Client

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    verbose=True,
)
structured_llm = llm.with_structured_output(
    AgentResponse
)  # it will create a new instance of the model capable to produce outputs in the desired structured format
client = Client()

react_prompt = client.pull_prompt("hwchase17/react")

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])

chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        input={
            "input": "search for 2 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
