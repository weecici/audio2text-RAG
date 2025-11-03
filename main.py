import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)

client = Cerebras(api_key=CEREBRAS_API_KEY)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important? Answer in brief.",
        }
    ],
    model="gpt-oss-120b",
)
print(chat_completion.choices[0].message.content)
