import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters  # For stdio-based MCP servers

print("\n=== Starting Oceanographic Data Assistant Crew ===\n")

# --- LLM Setup ---
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
)

# --- MCP Setup (SQLite DB server) ---
server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@antv/mcp-server-chart"
      ]
)

with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
    print(f"✅ Available MCP tools: {[tool.name for tool in mcp_tools]}")

    # --- Agents ---
    prompt_guard = Agent(
        role="Prompt Guard Agent",
        goal="Check if the user input is safe and relevant to oceanographic queries.",
        backstory="Strict filter that blocks unsafe prompts.",
        llm=llm,
        verbose=True,
        memory=True,
    )

    query_processor = Agent(
        role="Query Processor Agent",
        goal="Interpret safe user queries and do the appropriate visualizations using the mcp server tools.",
        backstory=(
            "You are an ocean data assistant who queries the ARGO database "
            "via MCP tools, do chart visualizations."
        ),
        llm=llm,
        verbose=True,
        memory=True,
        tools=mcp_tools,  # ✅ Attach MCP tools
    )

    output_formatter = Agent(
        role="Output Formatter Agent",
        goal="Format the final response into clean, structured text.",
        backstory="Ensures safe, user-friendly, and dashboard-ready responses.",
        llm=llm,
        verbose=True,
        memory=True,
    )

    # --- Tasks ---
    guard_task = Task(
        description=(
            "Check the input: {user_query}. "
            "If unsafe or irrelevant, respond ONLY with 'UNSAFE PROMPT'. "
            "If safe, respond with 'SAFE PROMPT'."
        ),
        name="guardrails",
        expected_output="Either 'SAFE PROMPT' or 'UNSAFE PROMPT'.",
        agent=prompt_guard,
    )

    process_task = Task(
        description=(
            "If guard output was 'SAFE PROMPT', process the query: {user_query}. "
            "Use the mcp-server-chart MCP tools to generate the appropriate visualizations. assume datapoints if needed "
            "If guard output was 'UNSAFE PROMPT', return 'BLOCKED'."
        ),
        name="processor",
        expected_output="A scientific summary or 'BLOCKED'.",
        agent=query_processor,
        tools=mcp_tools,  # ✅ Task can also call MCP tools
    )

    format_task = Task(
        description=(
            "Take the processor output and return a clean formatted message. "
            "If 'BLOCKED', say: '🚫 The input was unsafe and cannot be processed.' "
            "Otherwise, return the response as Markdown with sections."
        ),
        name="formatter",
        expected_output="A safe, user-friendly Markdown formatted answer.",
        agent=output_formatter,
    )

    # --- Crew ---
    crew = Crew(
        name="OceanCrew-turtle",
        agents=[prompt_guard, query_processor, output_formatter],
        tasks=[guard_task, process_task, format_task],
        process=Process.sequential,
        verbose=True,
        tracing=True,
        #memory=True,
    )

    # --- Run ---
    result = crew.kickoff(
        inputs={"user_query": input("Ask your question: ")}
    )

    print("\n=== Final Output ===\n")
    print(result)
