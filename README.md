# MCP Tool Server with LangGraph Agent

A complete **MCP-based tool ecosystem** consisting of:

* 🧰 **FastMCP Tool Server (Python)** exposing reusable tools
* 🤖 **LangGraph + LangChain Agent** that dynamically discovers and invokes MCP tools
* 🔄 **STDIO-based MCP client** implemented from scratch

This project demonstrates an **end-to-end agentic architecture** where an LLM reasons, selects tools dynamically, executes them via MCP, and responds to users interactively.

---

## 🚀 What This Project Achieves

✔ Builds a production-style **MCP Tool Server** using FastMCP
✔ Implements a **custom MCP client** using JSON-RPC over STDIO
✔ Dynamically converts MCP tools → LangChain tools at runtime
✔ Uses **LangGraph** to orchestrate reasoning + tool execution
✔ Uses **Groq LLM (LLaMA 3.3 70B)** for fast inference
✔ Fully testable using **MCP Inspector**

---

## 🧠 High-Level Architecture

```
User
 │
 ▼
LangGraph Agent (LLM)
 │
 ├─ Reasoning (LLM)
 ├─ Tool Selection
 │
 ▼
LangChain Tools (generated dynamically)
 │
 ▼
MCP Client (STDIO, JSON-RPC)
 │
 ▼
FastMCP Tool Server
 ├─ get_current_time
 ├─ get_weather
 ├─ internet_search
 ├─ math tools
```

This separation ensures **modularity, scalability, and clean responsibility boundaries**.

---

## 📂 Project Structure

```
Mcp-server/
│
├── agent.py        # LangGraph agent + MCP client
├── server.py       # FastMCP tool server
├── .env            # API keys (not committed)
├── requirements.txt
├── README.md
```

---

## 🛠️ Tools Exposed by MCP Server

### ⏰ Time Tool

* `get_current_time(timezone)`

### 🌦 Weather Tool

* `get_weather(city)` (OpenWeatherMap API)

### 🌐 Internet Search Tool

* `internet_search(query, num_results)` (Google Custom Search)

### ➗ Math Tools

* `add(a, b)`
* `subtract(a, b)`
* `multiply(a, b)`
* `divide(a, b)`
* `sqrt(a)`

All tools are registered using the `@mcp.tool()` decorator and auto-exposed via MCP schema.

---

## ⚙️ Prerequisites

* Python **3.10+**
* Groq API Key
* Google Custom Search API Key + CX
* OpenWeatherMap API Key

---

## 🔐 Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_search_engine_id
WEATHER_API_KEY=your_openweather_key
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/mcp-langgraph-agent.git
cd mcp-langgraph-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Running the System

### 1️⃣ Start the Agent (Server auto-starts)

```bash
python agent.py
```

The agent:

* Launches the MCP server as a subprocess
* Initializes MCP protocol
* Discovers tools dynamically
* Compiles LangGraph workflow

---

## 💬 Example Interaction

```
You: What is the current time in Asia/Kolkata?
AI: 2025-12-24T11:02:10+05:30

You: Weather in Hyderabad
AI: Weather in Hyderabad: clear sky, Temperature: 29°C, Humidity: 52%, Wind Speed: 3.1 m/s

You: What is 12 * 8?
AI: 96
```

---

## 🧩 Dynamic Tool Discovery (Key Feature)

The agent **does not hardcode tools**.

At runtime:

```python
tools = discover_tools(mcp_client)
model_with_tools = llm.bind_tools(tools)
```

Each MCP tool schema is converted into a LangChain-compatible tool **on the fly**, enabling:

* Plug-and-play tool servers
* Zero agent code changes when tools are added

---

## 🔁 LangGraph Workflow

* **agent node** → LLM reasoning
* **tools node** → Tool execution
* Conditional routing based on `tool_calls`

```python
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
```

This ensures **multi-step reasoning with tool usage**.

---

## 🧪 Testing MCP Server Independently

You can test `server.py` using **MCP Inspector**:

1. Select STDIO transport
2. Point to `server.py`
3. Invoke tools manually

Example:

```json
{
  "name": "get_weather",
  "arguments": { "city": "Bangalore" }
}
```

---

## 💡 Why STDIO Transport?

* Secure (no open ports)
* Ideal for agent subprocess execution
* Simple JSON-RPC protocol

HTTP transport can be added later if required.

---

## 🚧 Future Enhancements

* HTTP-based MCP transport
* Docker support
* Tool authentication
* Streaming responses
* Persistent memory

