"""
RAG orchestration logic
"""
import logging
from db.chroma_client import get_collection
from openai import OpenAI
from config import config
from memory.redis_window import RedisWindow

logger = logging.getLogger("rag_service")

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
        
        # Handle ChromaDB response format
        if docs and "documents" in docs and docs["documents"]:
            # docs["documents"] is a list of lists, we want the first list
            documents = docs["documents"][0] if docs["documents"][0] else []
            context = "\n".join(documents)
        else:
            context = ""
        
        if not context.strip():
            return "Sorry, I couldn't find relevant information to answer your question.", docs
        
        # Create a more detailed prompt for Bengali content
        prompt = f"""Answer the following question using the context below. If the question is in Bengali, answer in Bengali. If the question is in English, answer in English.

Context:
{context}

Question: {query}

Please provide a direct and accurate answer based only on the information provided in the context."""

        try:
            response = self.openai.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Your name is 'RAG4TenMS 🤖' . You are a helpful assistant developed by MUHIT KHAN (muhit.dev@gmai.com, https://muhit-khan.vercel.app, https://linkedin.com/in/muhit-khan) that answers very precisely questions based on provided context. Answer only if the information is grounded in the context or in this system message. If the question is in Bengali, respond in Bengali. If the question is in English, respond in English. "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            answer = response.choices[0].message.content or "Sorry, I couldn't generate an answer."
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "Sorry, I encountered an error while generating the answer."
        
        # Store conversation in memory
        try:
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", answer)
        except Exception as e:
            logger.warning(f"Could not store conversation in memory: {str(e)}")
        
        return answer, docs
