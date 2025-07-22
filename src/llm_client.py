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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
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
                
                content = response.choices[0].message.content
                return content.strip() if content else "Sorry, I couldn't generate a response."
            else:
                # Mock response for testing
                return self._generate_mock_response(query, language)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(query, language)
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt for the LLM"""
        if language == "bn":
            return """আপনি একজন বাংলা সাহিত্যের বিশেষজ্ঞ AI সহায়ক যিনি 'অপরিচিতা' গল্প সম্পর্কে গভীর জ্ঞান রাখেন। আপনার কাজ হল:

1. প্রদত্ত প্রসঙ্গ (context) থেকে প্রশ্নের সঠিক উত্তর খুঁজে বের করা
2. উত্তর সংক্ষিপ্ত, সুনির্দিষ্ট এবং সঠিক হতে হবে
3. চরিত্রের নাম, বয়স, সম্পর্ক ইত্যাদি সুনির্দিষ্ট তথ্যে ফোকাস করুন
4. যদি প্রসঙ্গে সরাসরি উত্তর না থাকে, সম্পর্কিত তথ্য থেকে উত্তর অনুমান করুন
5. সর্বদা বাংলায় উত্তর দিন

বিশেষ নির্দেশনা:
- অনুপমের সাথে সম্পর্কিত চরিত্রদের নাম সঠিকভাবে চিহ্নিত করুন
- বয়স, সময়কাল এর মতো সংখ্যাগত তথ্য খুবই গুরুত্বপূর্ণ
- শুধুমাত্র প্রদত্ত প্রসঙ্গের তথ্য ব্যবহার করুন, বাইরের জ্ঞান মিশাবেন না"""
        else:
            return """You are a Bengali literature expert AI assistant with deep knowledge of the story 'Oporichita'. Your tasks:

1. Find accurate answers from the provided context
2. Keep answers concise, specific, and accurate  
3. Focus on specific details like character names, ages, relationships
4. If direct answer isn't in context, infer from related information
5. Always respond in the same language as the question

Special instructions:
- Correctly identify character names related to Anupam
- Numerical information like ages and time periods are crucial
- Only use information from the provided context, don't mix external knowledge"""
    
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
        
        # Add context with better formatting
        if language == "bn":
            prompt_parts.extend([
                "\n=== প্রাসঙ্গিক তথ্য ===",
                context,
                f"\n=== প্রশ্ন ===",
                query,
                "\n=== উত্তর ===",
                "প্রাসঙ্গিক তথ্যের ভিত্তিতে সঠিক ও সংক্ষিপ্ত উত্তর:"
            ])
        else:
            prompt_parts.extend([
                "\n=== Relevant Information ===",
                context,
                f"\n=== Question ===", 
                query,
                "\n=== Answer ===",
                "Based on the relevant information, provide accurate and concise answer:"
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
