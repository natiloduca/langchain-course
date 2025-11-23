from dotenv import load_dotenv

load_dotenv()
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langsmith import Client

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    verbose=True,
)
client = Client()
# react_prompt = hub.pull("hwchase17/react") throw  error
react_prompt = client.pull_prompt("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("Hello Tavily")
    result = chain.invoke(
        input={
            "input": "search for the latest news about Tavily and summarize them in a concise manner."
            # "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details" // An output parsing error occurred
        }
    )
    print(result)


if __name__ == "__main__":
    main()
