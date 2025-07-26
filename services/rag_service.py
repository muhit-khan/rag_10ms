"""
RAG orchestration logic
"""
import logging
import sys
import os 
import time
from pathlib import Path
from db.chroma_client import get_collection
from openai import OpenAI
from config import config
from memory.redis_window import RedisWindow

# Configure logging
# Ensure logs directory exists
log_dir = Path("logs/rag_service")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"rag_service_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
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
                    {"role": "system", "content": """You are 'RAG4TenMS 🤖', an advanced document analysis specialist developed by MUHIT KHAN (muhit.dev@gmail.com, https://muhit-khan.vercel.app, https://linkedin.com/in/muhit-khan).

                    CORE DIRECTIVES:
                    • Provide ONLY information explicitly stated in the provided context
                    • Try to answer the question in one word if possible
                    • Maintain absolute factual accuracy - never extrapolate or assume
                    • For Bengali questions: respond in formal, literary Bengali
                    • For English questions: respond in clear, professional English
                    • Include exact quotes, names, numbers, and dates as they appear
                    • If information is unavailable or unclear, state this explicitly
                    • Prioritize precision over elaboration
                    
                    PROHIBITED ACTIONS:
                    • Do not supplement with external knowledge
                    • Do not make logical inferences beyond stated facts
                    • Do not provide generalized explanations
                    • Do not translate unless specifically requested"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
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
