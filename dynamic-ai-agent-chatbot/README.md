# Dynamic AI Agent Chatbot

A Colab-ready AI agent chatbot built with Python, Hugging Face Transformers, Qwen, and Gradio.

The agent can understand user intent, manage an in-memory schedule, detect scheduling conflicts, answer supported location/timezone questions, and run simple analytics.

## Features

- **Schedule management**
  - Create events
  - Update events
  - Delete events
  - Show the current schedule
  - Detect date/time conflicts

- **Natural date and time parsing**
  - `tomorrow`
  - `today`
  - `next Tuesday`
  - `3 pm`
  - `2:30 pm`
  - `noon`
  - `midnight`

- **Consistent date display**
  - All user-facing schedule dates are shown as `DD/MM/YYYY HH:MM`.

- **Location and timezone lookup**
  - Uses a local `LOCATION_DB`, so no paid API key is required.

- **Simple analytics**
  - Count
  - Sum
  - Average
  - Median
  - Minimum
  - Maximum

- **Gradio interface**
  - Runs in Google Colab
  - Uses tuple-based chat history for broad Gradio compatibility

## Project Structure

```text
dynamic-ai-agent-chatbot/
├── dynamic_ai_agent.py
├── dynamic_ai_agent_colab.ipynb
├── requirements.txt
├── README.md
├── demo_scenario.md
├── .gitignore
└── assets/
    └── .gitkeep
```

## How to Run in Google Colab

1. Open `dynamic_ai_agent_colab.ipynb` in Google Colab.
2. Run the installation cell.
3. Run all code cells.
4. Run:

```python
launch_ui(share=True)
```

5. Open the Gradio link that appears below the cell.

## How to Run Locally

Install the requirements:

```bash
pip install -r requirements.txt
```

Launch the Gradio UI:

```python
from dynamic_ai_agent import launch_ui

launch_ui(share=True)
```

You can also run the terminal loop:

```python
from dynamic_ai_agent import interactive_loop

interactive_loop()
```

## Demo Prompts

```text
Schedule a team meeting tomorrow at 10 am
Schedule a project review tomorrow at 10 am
Show my schedule
Move the team meeting to next Friday at 2:30 pm
What is the current local time in Paris?
Calculate the average and maximum for 10, 20, 30, 40
```

## Notes

- Schedule data is stored in memory only.
- Restarting the notebook/runtime clears the schedule.
- The Qwen model is loaded lazily, so it only loads when needed for general LLM responses.
- The rule-based layer handles the main demo features to keep the project reliable during presentations.
