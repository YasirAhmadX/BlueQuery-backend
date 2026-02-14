from crewai import LLM
import os
from crewai import Agent, Task, Crew, Process


print("\n=== Starting Oceanographic Data Assistant Crew ===\n")
# Mock: API Keys (replace with real if using LLMs)
# os.environ["GEMINI_API_KEY"] = "Api key"

llm = LLM(
    model="ollama/qwen2.5:0.5b",
    temperature=0.7,
    api_base="http://localhost:11434",
    api_key="dummy"
)

# --- Define Agents ---
prompt_guard = Agent(
    role="Prompt Guard Agent",
    goal="Check if the user input is safe and relevant to oceanographic queries.",
    backstory=(
        "You are a strict filter that decides if a query should be processed. "
        "If unsafe, you stop the pipeline by flagging it."
    ),
    llm=llm,
    verbose=True,
    memory=True,
)

query_processor = Agent(
    role="Query Processor Agent",
    goal="Interpret safe user queries and generate useful scientific responses.",
    backstory=(
        "You are an ocean data assistant who knows how to fetch, analyze, "
        "and summarize ARGO float data."
    ),
    llm=llm,
    verbose=True,
    memory=True,
)

output_formatter = Agent(
    role="Output Formatter Agent",
    goal="Sanitize and format the final response into clean, structured text.",
    backstory=(
        "You make the final response user-friendly, well-formatted, and safe "
        "for display in dashboards or chat."
    ),
    llm=llm,
    verbose=True,
    memory=True,
)

# --- Define Tasks ---
guard_task = Task(
    description=(
        "Check the input: {user_query}. "
        "If the query is unsafe or irrelevant, respond ONLY with 'UNSAFE PROMPT'. "
        "If safe, respond with 'SAFE PROMPT'."
    ),
    name = "guardrails",
    expected_output="Either 'SAFE PROMPT' or 'UNSAFE PROMPT'.",
    agent=prompt_guard,
)

process_task = Task(
    description=(
        "If the guard output was 'SAFE PROMPT', process the user query: {user_query}. "
        "Return a mock processed output (e.g., salinity profile, trajectory, etc.). "
        "If guard output was 'UNSAFE PROMPT', just return 'BLOCKED'."
    ),
    name="processor",
    expected_output="A scientific summary or 'BLOCKED'.",
    agent=query_processor,
)

format_task = Task(
    description=(
        "Take the processor output and return a clean formatted message. "
        "If it was 'BLOCKED', say: '🚫 The input was unsafe and cannot be processed.' "
        "Otherwise, return the response as Markdown with sections."
    ),
    name="formatter",
    expected_output="A safe, user-friendly Markdown formatted answer.",
    agent=output_formatter,
)

# --- Assemble Crew ---
crew = Crew(
    name = "OceanCrew",
    agents=[prompt_guard, query_processor, output_formatter],
    tasks=[guard_task, process_task, format_task],
    process=Process.sequential,
    verbose=True,
    tracing=True,
    #memory=True,
)

# --- Run Example ---
result = crew.kickoff(
    inputs={"user_query": "Show me salinity profiles near the equator in March 2023"}
)

print("\n=== Final Output ===\n")
print(result)
