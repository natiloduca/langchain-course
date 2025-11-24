# chapter 4
from dotenv import load_dotenv

load_dotenv()
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
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
client = Client()

react_prompt = client.pull_prompt("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    try:
        result = chain.invoke(
            input={
                "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
            }
        )
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
