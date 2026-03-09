# Deep Research Agent

<div align="center">
    <a href="https://your-domain.streamlit.app">
        <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" width="200" alt="Open in Streamlit">
    </a>
</div>

An autonomous AI agent that performs deep web research, reasoning, and reporting. Built with **Google Gemini**, **LangGraph**, and **Streamlit**.

![Agent Architecture](agent_graph.png)

## 🚀 Features
- **Cyclic Agentic Workflow:** Uses LangGraph to create a loop of Reasoning → Acting → Observing.
- **Deep Web Search:** Integrated with Tavily API for real-time, cited information retrieval.
- **Long-term Memory:** Maintains conversational context across multiple turns using persistence.
- **Adaptive RL Layer:** Learns response strategy (concise, detailed, citations, step-by-step) from rewards.
- **Interaction Memory:** Stores a structured learning record after every response, including reward, lessons learned, and future adjustment.
- **Visual Interface:** Clean, interactive frontend built with Streamlit.
- **Professional UI:** Displays "Thinking..." steps and rich markdown responses.
- **Interactive Feedback:** Rate each answer with 👍 / 👎 to immediately update the policy.

## 🛠️ Tech Stack
- **Brain:** Google Gemini 2.5 Flash
- **Orchestration:** LangGraph (State Machines)
- **Tools:** Tavily Search API
- **Frontend:** Streamlit
- **Language:** Python 3.10+

## ⚙️ Setup & Run

1. **Clone the repository**
   ```bash
   

    ```

2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Configure API Keys**
Create a `.env` file in the root directory and add your keys (never share this file):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_key_here

```


4. **Run the App**
```bash
streamlit run app.py

```



## 🧠 Architecture

The agent follows a **ReAct** (Reason + Act) pattern:

1. **Input:** User asks a question (e.g., "What is the stock price of Apple?").
2. **Decision:** The agent decides if it needs external information.
3. **Action:** If yes, it queries the Tavily Search API.
4. **Observation:** It reads the search results.
5. **Loop:** It determines if it has enough info to answer or needs to search again.
6. **Output:** Final synthesized answer delivered to the Streamlit UI .

## 🔁 Reinforcement Learning Loop

Each assistant turn now includes a lightweight policy-selection and feedback loop:

1. The user prompt is classified into a state (`realtime`, `comparison`, `coding`, etc.).
2. The policy picks a response action (`concise`, `detailed`, `with_citations`, `step_by_step`).
3. The assistant generates the response using that strategy.
4. The response is stored as a neutral learning record until user reaction is available.
5. Explicit feedback or a clearly positive or negative follow-up updates Q-values immediately.

The learned policy is persisted in `rl_policy.json` so the assistant improves over time.

## 🧾 Post-Conversation Learning

After every assistant response, the app creates a structured memory entry with this shape:

```json
{
    "conversation_summary": "...",
    "user_intent": "...",
    "agent_response_quality": "good | bad | neutral",
    "reward": 1,
    "lessons_learned": "...",
    "future_adjustment": "..."
}
```

Rewards follow these rules:

1. Positive reactions or successful completion map to `+1`.
2. Confusion, dissatisfaction, or clear correction map to `-1`.
3. Unclear outcomes stay `0` until stronger feedback appears.

Structured learning history is persisted in `interaction_memory.json`.

## Provider Notes

The app now uses Google Gemini with the `gemini-2.5-flash` model, and expects `GOOGLE_API_KEY`.

Tavily is optional. If `TAVILY_API_KEY` is not set, the app still runs, but it answers without live web search.

