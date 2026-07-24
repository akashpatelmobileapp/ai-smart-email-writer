from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool  # ← correct import
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient


llm = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_attachments(topic: str) -> str:
    """Search for relevant attachment suggestions for a given email topic."""
    response = tavily_client.search(        # ← instance, not class
        query=f"relevant documents attachments for {topic} email",
        max_results=5,
    )
    results = response.get("results", [])
    if not results:
        return "No attachment suggestions found."

    suggestions = []
    for i, r in enumerate(results):
        suggestions.append(f"{i+1}. {r['title']}\n   {r['content'][:100]}...")

    return "\n\n".join(suggestions)

draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an email writing assistant."),
    ("human", "Write a {tone} email about: {topic}")
])

# Grammar — takes email
grammar_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grammar correction tool. Fix and improve this email."),
    ("human", "{email}")
])

# Shorten — takes email
shorten_prompt = ChatPromptTemplate.from_messages([
    ("system", "Shorten this email to under 150 words. Keep the full meaning."),
    ("human", "{email}")
])


draft_chain = draft_prompt | llm | parser
grammar_chain = grammar_prompt | llm | parser
shorten_chain = shorten_prompt | llm | parser

email_chain = (
    draft_chain
    | RunnableLambda(lambda x: {"email": x})   # wrap string → dict for next prompt
    | grammar_chain
    | RunnableLambda(lambda x: {"email": x})   # wrap again
    | shorten_chain
)

# Step 1: Take input ONCE at the start
topic = input("Topic : ").strip()
tone  = input("Tone  (formal/casual/friendly): ").strip()

# Step 2: Chain runs IMMEDIATELY — no user involvement
print("\n[bold yellow]⚙ Drafting → Fixing Grammar → Shortening...[/bold yellow]\n")
final_email = email_chain.invoke({"topic": topic, "tone": tone})

# Step 3: Show the result
print("[bold cyan]─── Final Email ────────────────────────────[/bold cyan]")
print(final_email)
print("[bold cyan]────────────────────────────────────────────[/bold cyan]\n")

# Step 4: ONLY NOW does agent loop start — for follow-up only
print("Commands:")
print("  [bold]'add attachment suggestion'[/bold] → search relevant attachments")
print("  [bold]'exit'[/bold] → quit\n")
# ── Agent setup ───────────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=[search_attachments],
)

# ── Agent loop ────────────────────────────────────────────
print("  'add attachment suggestion' → search relevant attachments")
print("  'exit' → quit\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    # ✅ create_agent uses messages format, not {"input": ...}
    response = agent.invoke({
        "messages": [
            SystemMessage(content="""You are an email assistant.
When user asks for attachment suggestions, call the search_attachments tool.
Present results as a clean numbered list with 📎 header.
Only call the tool when explicitly asked."""),
            HumanMessage(content=f"Email topic: {topic}\n\nUser request: {user_input}")
        ]
    })

    # ✅ output is in last message, not response["output"]
    print(f"\nAI: {response['messages'][-1].content}\n")

    # Topic : project deadline extension
# Tone  : formal