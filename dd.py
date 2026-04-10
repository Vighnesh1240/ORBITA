from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

models = list(client.models.list())

print("Total models:", len(models))

for m in models:
    print(m.name)
# Use a supported model from the list above
response = client.models.generate_content(
    model="models/gemini-2.5-pro",
    contents="Say hello"
)

print(response.text)   