import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Generator:
    def __init__(self, model="meta-llama/llama-3-8b-instruct"):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key is None:
            raise ValueError("❌ OPENROUTER_API_KEY not found")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model = model

    def build_context(self, retrieved_docs):
        """
        Builds context and creates consistent source mapping
        """
        context = ""
        source_map = {}
        source_counter = 1

        for doc in retrieved_docs:
            source = doc["metadata"].get("source", "unknown")

            # Skip graph in citation numbering
            if source != "graph":
                if source not in source_map:
                    source_map[source] = source_counter
                    source_counter += 1

                source_id = source_map[source]
                context += f"[Source {source_id}]\n{doc['text']}\n\n"
            else:
                context += f"[Graph]\n{doc['text']}\n\n"

        return context, source_map

    def generate_answer(self, query, retrieved_docs):
        """
        Generate answer + return source mapping
        """

        context, source_map = self.build_context(retrieved_docs)

        prompt = f"""
You are an AI assistant answering questions based ONLY on the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Provide a detailed and well-structured answer
- Use citations like [Source 1], [Source 2] wherever relevant
- Do NOT make up citations
- Use only the given sources
- Structure answer as:
    1. Definition / Overview
    2. Key Explanation
    3. Important Points (if applicable)
- If the answer is not present, say: "I don't know based on the provided documents"

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        answer = response.choices[0].message.content.strip()

        return answer, source_map