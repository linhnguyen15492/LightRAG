from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("UPSTAGE_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1/solar")

stream = client.chat.completions.create(
    model="solar-mini",
    messages=[
        {
            "role": "user",
            "content": "Question 1: What metrics can be utilized to assess reader engagement with the poetry selected from the dataset for the anthology?",
        }
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Use with stream=False
# print(stream.choices[0].message.content)
