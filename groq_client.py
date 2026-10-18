
import os

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from your .env file."
    )

client = AsyncGroq(
    api_key=API_KEY,
    timeout=30.0
)

SYSTEM_PROMPT = """
You are Chat Scholar, an academic AI assistant.

Your responsibilities:
1. Explain academic concepts clearly.
2. Answer follow-up questions using conversation context.
3. Provide examples when helpful.
4. Be accurate and acknowledge uncertainty.
5. Help students understand complex topics.
"""


async def ask_groq(question: str, history: list) -> str:
    """
    Send conversation history and the new question
    to Groq.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Add previous conversation messages
    messages.extend(history)

    # Add the latest user question
    messages.append({
        "role": "user",
        "content": question
    })

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024
    )

    return completion.choices[0].message.content or ""
