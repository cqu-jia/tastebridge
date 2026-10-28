"""List models available on Nebius Token Factory (look for nvidia/ Nemotron ones)."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL"), api_key=os.environ["NEBIUS_API_KEY"])
for m in client.models.list():
    print(m.id)
