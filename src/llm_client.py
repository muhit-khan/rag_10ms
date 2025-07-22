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
                    temperature=0.0,  # Zero temperature for factual responses
                    max_tokens=150,   # Shorter responses to prevent hallucination
                    top_p=0.1,        # Very focused responses
                    frequency_penalty=0.0,
                    presence_penalty=0.0
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
            return """আপনি একটি কঠোর নিয়মাবলী অনুসরণকারী AI সহায়ক। আপনার একমাত্র কাজ হল:

CRITICAL RULES (অবশ্যই মেনে চলুন):
1. শুধুমাত্র প্রদত্ত CONTEXT থেকে সরাসরি উদ্ধৃত তথ্য ব্যবহার করুন
2. কোনো অনুমান করবেন না, কোনো অতিরিক্ত জ্ঞান যোগ করবেন না
3. Context-এ উত্তর না থাকলে বলুন "প্রদত্ত তথ্যে এই প্রশ্নের উত্তর নেই"
4. উত্তর সংক্ষিপ্ত রাখুন (সর্বোচ্চ ২-৩ বাক্য)
5. নাম, সংখ্যা, বয়স হুবহু Context থেকে কপি করুন

FORBIDDEN (নিষিদ্ধ):
- Context-এর বাইরের কোনো তথ্য
- অনুমান বা ধারণা
- "সম্ভবত", "মনে হয়", "হতে পারে" জাতীয় শব্দ
- দীর্ঘ ব্যাখ্যা

FORMAT: শুধু সরাসরি উত্তর দিন, অতিরিক্ত কথা নয়।"""
        else:
            return """You are a strictly rule-following AI assistant. Your ONLY job is:

CRITICAL RULES (Must follow):
1. Use ONLY information directly quoted from the provided CONTEXT
2. NO assumptions, NO additional knowledge
3. If answer not in context, say "The answer is not available in the provided information"
4. Keep answers brief (maximum 2-3 sentences)
5. Copy names, numbers, ages exactly from context

FORBIDDEN:
- Any information outside the context
- Assumptions or guesses
- Words like "probably", "seems", "might be"
- Long explanations

FORMAT: Give direct answer only, no extra commentary."""
    
    def _build_user_prompt(self, 
                          query: str, 
                          context: str, 
                          conversation_history: Optional[List[Dict]] = None,
                          language: str = "bn") -> str:
        """Build user prompt with context and history"""
        
        prompt_parts = []
        
        # Add context with strict instructions
        if language == "bn":
            prompt_parts.extend([
                "=== CONTEXT (একমাত্র তথ্যের উৎস) ===",
                context,
                "\n=== INSTRUCTION ===",
                "উপরের CONTEXT থেকে নিচের প্রশ্নের উত্তর খুঁজুন। Context-এ না থাকলে 'প্রদত্ত তথ্যে এই প্রশ্নের উত্তর নেই' বলুন।",
                f"\n=== QUESTION ===",
                query,
                "\n=== ANSWER (Context থেকে সরাসরি) ===",
            ])
        else:
            prompt_parts.extend([
                "=== CONTEXT (Only source of information) ===",
                context,
                "\n=== INSTRUCTION ===",
                "Find answer to the question below from the CONTEXT above. If not in context, say 'The answer is not available in the provided information.'",
                f"\n=== QUESTION ===", 
                query,
                "\n=== ANSWER (Direct from context) ===",
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
