from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manage short-term conversation history"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations = {}  # session_id -> conversation history
    
    def add_exchange(self, session_id: str, user_query: str, assistant_response: str):
        """Add a query-response pair to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "assistant_response": assistant_response
        }
        
        self.conversations[session_id].append(exchange)
        
        # Keep only recent history
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def get_recent_queries(self, session_id: str, count: int = 3) -> List[str]:
        """Get recent user queries for context"""
        history = self.get_history(session_id)
        recent_queries = [exchange["user_query"] for exchange in history[-count:]]
        return recent_queries
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]


class LLMClient:
    """Client for Language Model interaction"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the LLM client"""
        try:
            if self.api_key:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model}")
            else:
                logger.warning("No API key provided - using mock responses")
                
        except ImportError:
            logger.error("OpenAI package not installed")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
    
    def generate_response(self, 
                         query: str, 
                         context: str, 
                         conversation_history: Optional[List[Dict]] = None,
                         language: str = "bn") -> str:
        """Generate response using retrieved context"""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(language)
        
        # Build user prompt with context
        user_prompt = self._build_user_prompt(query, context, conversation_history, language)
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                return response.choices[0].message.content.strip()
            else:
                # Mock response for testing
                return self._generate_mock_response(query, language)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(query, language)
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt for the LLM"""
        if language == "bn":
            return """আপনি একজন সহায়ক AI সহায়ক যিনি বাংলা সাহিত্য বিশেষজ্ঞ। আপনার কাজ হল:

1. প্রদত্ত প্রসঙ্গ থেকে প্রশ্নের উত্তর দেওয়া
2. উত্তর সংক্ষিপ্ত এবং সঠিক হতে হবে
3. যদি প্রসঙ্গে উত্তর না থাকে, তাহলে "আমি নিশ্চিত নই" বলুন
4. সর্বদা বাংলায় উত্তর দিন

মনে রাখবেন: শুধুমাত্র প্রদত্ত প্রসঙ্গের তথ্য ব্যবহার করুন।"""
        else:
            return """You are a helpful AI assistant specialized in Bengali literature. Your tasks:

1. Answer questions based on the provided context
2. Keep answers concise and accurate
3. If the answer is not in the context, say "I'm not certain"
4. Always respond in the same language as the question

Remember: Only use information from the provided context."""
    
    def _build_user_prompt(self, 
                          query: str, 
                          context: str, 
                          conversation_history: Optional[List[Dict]] = None,
                          language: str = "bn") -> str:
        """Build user prompt with context and history"""
        
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            if language == "bn":
                prompt_parts.append("পূর্ববর্তী কথোপকথন:")
            else:
                prompt_parts.append("Previous conversation:")
            
            for exchange in conversation_history[-2:]:  # Last 2 exchanges
                prompt_parts.append(f"প্রশ্ন: {exchange['user_query']}")
                prompt_parts.append(f"উত্তর: {exchange['assistant_response']}")
        
        # Add context
        if language == "bn":
            prompt_parts.extend([
                "\nপ্রসঙ্গ তথ্য:",
                context,
                f"\nপ্রশ্ন: {query}",
                "\nউত্তর:"
            ])
        else:
            prompt_parts.extend([
                "\nContext information:",
                context,
                f"\nQuestion: {query}",
                "\nAnswer:"
            ])
        
        return "\n".join(prompt_parts)
    
    def _generate_mock_response(self, query: str, language: str) -> str:
        """Generate mock response for testing"""
        if language == "bn":
            return f"[মক উত্তর] আপনার প্রশ্ন '{query[:50]}...' এর জন্য একটি নমুনা উত্তর।"
        else:
            return f"[Mock Response] Sample answer for your question '{query[:50]}...'"
    
    def _generate_fallback_response(self, query: str, language: str) -> str:
        """Generate fallback response when LLM fails"""
        if language == "bn":
            return "দুঃখিত, আমি এই মুহূর্তে আপনার প্রশ্নের উত্তর দিতে পারছি না।"
        else:
            return "Sorry, I'm unable to answer your question at the moment."
