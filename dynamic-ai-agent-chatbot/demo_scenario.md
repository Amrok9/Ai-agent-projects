# Demo Scenario

Use this scenario when recording a LinkedIn demo or presenting the project.

## 1. Create an event

```text
Schedule a team meeting tomorrow at 10 am
```

Expected behavior: the agent creates a schedule event and displays the timestamp as `DD/MM/YYYY HH:MM`.

## 2. Demonstrate conflict detection

```text
Schedule a project review tomorrow at 10 am
```

Expected behavior: the agent detects that another event already exists at the same date and time.

## 3. Show the current schedule

```text
Show my schedule
```

Expected behavior: the agent lists all saved events from the current runtime session.

## 4. Update an event

```text
Move the team meeting to next Friday at 2:30 pm
```

Expected behavior: the agent updates the existing event and converts `2:30 pm` to `14:30`.

## 5. Ask a timezone question

```text
What is the current local time in Paris?
```

Expected behavior: the agent returns the country, timezone, and current local time.

## 6. Run analytics

```text
Calculate the average and maximum for 10, 20, 30, 40
```

Expected behavior: the agent returns count, sum, average, median, minimum, and maximum.

## Suggested LinkedIn Caption

Built a Dynamic AI Agent Chatbot using Python, Hugging Face Transformers, Qwen, and Gradio.

The agent can manage an in-memory schedule, parse natural dates and times, detect scheduling conflicts, answer timezone questions, and run simple analytics. This project helped me practice agent design, rule-based parsing, LLM integration, and interactive UI development.
