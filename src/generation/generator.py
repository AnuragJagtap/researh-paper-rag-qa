import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Generator:
    def __init__(self, model="meta-llama/llama-3-8b-instruct"):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key is None:
            raise ValueError("❌ OPENROUTER_API_KEY not found in environment variables")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model = model

    def generate_answer(self, query, retrieved_docs):
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])

        prompt = f"""
You are an AI assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Answer ONLY from the context
- If answer is not present, say "I don't know based on the provided documents"
- Be concise and clear

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()