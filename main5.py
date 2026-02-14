## this code adapts devsan-mcp to fastapi endpoint for frontend integration
import logging

# Suppress ChromaDB / Posthog noisy logs
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.ERROR)

# Optional: suppress everything except CrewAI verbose
logging.basicConfig(level=logging.ERROR)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# crewai imports (keep the same as your original script)
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters  # For stdio-based MCP servers

app = FastAPI(title="Oceanographic Data Assistant API")

# Request model
class QueryRequest(BaseModel):
    query: str

# Make DB path configurable via env var (fallback to your original path)
ARGO_DB_PATH = os.environ.get(
    "ARGO_DB_PATH",
    "C:/Users/YasirAhmd/Downloads/agro_db_5_floats.db"
)

# Pre-create LLM instance (shared)
llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    temperature=0.7,
)

server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@executeautomation/database-server",
            "C:/Users/YasirAhmd/Downloads/agro_db_5_floats.db"
        ],
        env={**os.environ},
    )
@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Endpoint that:
    - runs the prompt guard agent,
    - runs the query processor (with MCP tools),
    - runs the output formatter,
    - returns the final formatted result.
    """
    user_query = request.query

    # --- MCP Setup (make sure `npx @executeautomation/database-server` is available) ---
    

    try:
        with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
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
                goal="Interpret safe user queries and answer those queries.FOr normal queries you have to answer without tool call, If the queries are related to data then only fetch/analyze ARGO float data using the SQLite MCP tools.",
                backstory=(
                    "You are an ocean data assistant who queries the ARGO database "
                    "via MCP tools, analyzes the results, and produces summaries."
                ),
                llm=llm,
                verbose=True,
                memory=True,
                tools=mcp_tools,  # attach MCP tools
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
                    "If unsafe or irrelevant to ocean and argo projects, respond ONLY with 'UNSAFE PROMPT'. "
                    "If safe, respond with 'SAFE PROMPT'."
                ),
                name="guardrails",
                expected_output="Either 'SAFE PROMPT' or 'UNSAFE PROMPT'.",
                agent=prompt_guard,
                verbose=True,
            )

            process_task = Task(
                description=(
                    "If guard output was 'SAFE PROMPT', process the query: {user_query}. "
                    "If the prompt is a normal question, answer it directly without using any tool. "
                    "Use the SQLite MCP tools to run SQL queries against the ARGO DB. "
                    "Return a scientific summary (salinity profile, trajectory, etc.). "
                    "If guard output was 'UNSAFE PROMPT', return 'BLOCKED'."
                ),
                name="processor",
                expected_output="A scientific summary or 'BLOCKED'.",
                agent=query_processor,
                tools=mcp_tools,
                verbose=True,
            )

            format_task = Task(
                description=(
                    "Take the processor output and return a clean formatted message."
                    "If 'BLOCKED', say: '🚫 The input was unsafe and cannot be processed.' "
                    "Otherwise, return the response as clearly readable Markdown with sections."
                    "use linebreaks, blockquotes, bold letters and bullet points"
                ),
                name="formatter",
                expected_output="return only answer, user-friendly Markdown formatted answer.",
                agent=output_formatter,
                verbose=True,
            )

            # --- Crew ---
            crew = Crew(
                name="Argonaut School",
                agents=[prompt_guard, query_processor, output_formatter],
                tasks=[guard_task, process_task, format_task],
                #agents=[prompt_guard, query_processor],
                #tasks=[guard_task, process_task],
                process=Process.sequential,
                verbose=True,
                tracing=True,
                memory = False,
            )

            # --- Run crew with the user's query ---
            result = crew.kickoff(inputs={"user_query": user_query})

            # Return the crew output
            # result may be a dict-like or string depending on Crew API; adapt if needed
            return {"query": user_query, "result": result}

    except Exception as e:
        # Provide helpful HTTP error
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
