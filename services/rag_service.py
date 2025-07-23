"""
RAG orchestration logic
"""
from db.chroma_client import get_collection
from openai import OpenAI
from config import config
from memory.redis_window import RedisWindow

class RAGService:
    def __init__(self, user_id: str):
        self.collection = get_collection()
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.memory = RedisWindow(user_id)

    def embed_query(self, query: str):
        response = self.openai.embeddings.create(
            input=query,
            model=config.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def search(self, query: str, k: int = 5):
        embedding = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={}
        )
        return results

    def generate_answer(self, query: str):
        docs = self.search(query)
        context = "\n".join([doc["document"] for doc in docs["documents"][0]])
        prompt = f"Answer the following question using the context below.\nContext:\n{context}\nQuestion: {query}"
        response = self.openai.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": "Answer only if grounded."},
                      {"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)
        return answer, docs
