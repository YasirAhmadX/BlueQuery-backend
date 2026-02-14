import os
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# crewai imports
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters  # For stdio-based MCP servers

app = FastAPI(title="Oceanographic Data Assistant API (persistent MCP)")

class QueryRequest(BaseModel):
    query: str

ARGO_DB_PATH = os.environ.get(
    "ARGO_DB_PATH",
    "C:/Users/YasirAhmd/Downloads/agro_db_5_floats.db"
)

# GLOBALS to be initialized at startup
mcp_tools: MCPServerAdapter = None
crew: Crew = None
executor: ThreadPoolExecutor = None
mcp_lock: asyncio.Semaphore = None

# Tunables
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "4"))
MCP_CONNECT_TIMEOUT = int(os.environ.get("MCP_CONNECT_TIMEOUT", "20"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))
THREAD_POOL_WORKERS = int(os.environ.get("THREAD_POOL_WORKERS", "4"))

# Reusable LLM instance (keep)
llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.7)

# Optional: simple in-memory LRU cache for identical queries (speeds repeated queries)
# You can tune cache size depending on memory.
@lru_cache(maxsize=256)
def _cached_result_for_query(query_text: str):
    # NOTE: This wrapper will not be used to call the crew directly (since kickoff is async)
    # But we keep this so the request handler can use it to short-circuit repeated queries.
    return None  # placeholder — we'll call crew normally, then set cache manually below.

@app.on_event("startup")
async def startup_event():
    global mcp_tools, crew, executor, mcp_lock

    # Create a thread pool for blocking kickoff calls
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

    # Create semaphore to limit concurrent requests to the MCP + Crew
    mcp_lock = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # --- Start persistent MCP server connection once ---
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@executeautomation/database-server",
            ARGO_DB_PATH
        ],
        env={**os.environ},
    )

    # create adapter once (connect_timeout small, since we persist)
    mcp_tools = MCPServerAdapter(server_params, connect_timeout=MCP_CONNECT_TIMEOUT)
    # If MCPServerAdapter needs a connect() call, call it here. (API dependent)
    # e.g. mcp_tools.connect()

    # --- Agents & Tasks (constructed once) ---
    # Make memory=False and verbose=False in production for speed unless you need them.
    prompt_guard = Agent(
        role="Prompt Guard Agent",
        goal="Check if the user input is safe and relevant to oceanographic queries.",
        backstory="Strict filter that blocks unsafe prompts.",
        llm=llm,
        verbose=False,
        memory=False,
    )

    query_processor = Agent(
        role="Query Processor Agent",
        goal="Interpret safe user queries and fetch/analyze ARGO float data using the SQLite MCP tools.",
        backstory=(
            "You are an ocean data assistant who queries the ARGO database "
            "via MCP tools, analyzes the results, and produces summaries."
        ),
        llm=llm,
        verbose=False,
        memory=False,
        tools=mcp_tools,
    )

    output_formatter = Agent(
        role="Output Formatter Agent",
        goal="Format the final response into clean, structured text.",
        backstory="Ensures safe, user-friendly, and dashboard-ready responses.",
        llm=llm,
        verbose=False,
        memory=False,
    )

    guard_task = Task(
        description=(
            "Check the input: {user_query}. "
            "If unsafe or irrelevant, respond ONLY with 'UNSAFE PROMPT'. "
            "If safe, respond with 'SAFE PROMPT'."
        ),
        name="guardrails",
        expected_output="Either 'SAFE PROMPT' or 'UNSAFE PROMPT'.",
        agent=prompt_guard,
        verbose=False,
    )

    process_task = Task(
        description=(
            "If guard output was 'SAFE PROMPT', process the query: {user_query}. "
            "Use the SQLite MCP tools to run SQL queries against the ARGO DB. "
            "Return a scientific summary (salinity profile, trajectory, etc.). "
            "If guard output was 'UNSAFE PROMPT', return 'BLOCKED'."
        ),
        name="processor",
        expected_output="A scientific summary or 'BLOCKED'.",
        agent=query_processor,
        tools=mcp_tools,
        verbose=False,
    )

    format_task = Task(
        description=(
            "Take the processor output and return a clean formatted message. Format tabular data in Markdown tables. "
            "If 'BLOCKED', say: '🚫 The input was unsafe and cannot be processed.' "
            "Otherwise, return the response as Markdown with sections."
        ),
        name="formatter",
        expected_output="A safe, user-friendly Markdown formatted answer.",
        agent=output_formatter,
        verbose=False,
    )

    # --- Crew (single instance reused) ---
    crew = Crew(
        name="OceanCrew-turtle",
        agents=[prompt_guard, query_processor, output_formatter],
        tasks=[guard_task, process_task, format_task],
        process=Process.sequential,  # or parallel if your tasks are independent
        verbose=False,
        tracing=False,
    )

@app.on_event("shutdown")
async def shutdown_event():
    global mcp_tools, executor
    try:
        if mcp_tools is not None:
            # If adapter has a close method, call it.
            try:
                mcp_tools.close()
            except Exception:
                # fallback: if context manager, ensure any cleanup occurs here
                pass
            mcp_tools = None
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    - Uses shared MCP + Crew instances created at startup.
    - Limits concurrency with semaphore.
    - Offloads blocking kickoff to thread pool executor.
    """
    global crew, mcp_tools

    if crew is None or mcp_tools is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    # Optional: quick LRU cache check (simple exact-text cache)
    # We keep a small manual cache mapping query -> result
    # (You can replace this with cachetools.TTLCache if you want TTL)
    cache_key = f"q:{user_query}"
    # if cached, return quickly (very simple - you may persist this)
    # NOTE: _cached_store is a simple global dict stored in-memory
    if hasattr(process_query, "_cached_store") and cache_key in process_query._cached_store:
        return {"query": user_query, "result": process_query._cached_store[cache_key], "cached": True}

    # Acquire semaphore to limit simultaneous use of MCP/Crew
    async with mcp_lock:
        try:
            # kickoff is likely blocking; run it in executor
            # Use run_in_executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            kickoff_call = lambda: crew.kickoff(inputs={"user_query": user_query})
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, kickoff_call),
                    timeout=REQUEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail=f"Processing timed out after {REQUEST_TIMEOUT_SECONDS} seconds.")

            # Save in simple cache
            if not hasattr(process_query, "_cached_store"):
                process_query._cached_store = {}
            # Keep small cache size
            if len(process_query._cached_store) > 512:
                # naive eviction: clear (replace with LRU cache impl if desired)
                process_query._cached_store.clear()
            process_query._cached_store[cache_key] = result

            return {"query": user_query, "result": result, "cached": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
