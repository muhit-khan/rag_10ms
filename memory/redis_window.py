"""
Conversation buffer logic
"""
import logging
from typing import List, Optional, Tuple, Any
from db.redis_client import get_redis_client
from config import config

logger = logging.getLogger("redis_window")

class RedisWindow:
    def __init__(self, user_id: str, window_size: int = 4):
        self.user_id = user_id
        self.window_size = window_size
        self.key = f"chat:{user_id}"
        self.fallback_memory: List[str] = []  # In-memory fallback
        self.client: Optional[Any] = None
        
        try:
            self.client = get_redis_client()
            # Test connection
            self.client.ping()
            self.redis_available = True
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory fallback: {str(e)}")
            self.client = None
            self.redis_available = False

    def add_message(self, role: str, content: str):
        message = f"{role}:{content}"
        
        if self.redis_available:
            try:
                self.client.rpush(self.key, message)
                self.client.ltrim(self.key, -self.window_size, -1)
                return
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {str(e)}")
                self.redis_available = False
        
        # Fallback to in-memory storage
        self.fallback_memory.append(message)
        if len(self.fallback_memory) > self.window_size:
            self.fallback_memory = self.fallback_memory[-self.window_size:]

    def get_window(self):
        if self.redis_available:
            try:
                messages = self.client.lrange(self.key, 0, -1)
                return [msg.split(":", 1) for msg in messages]
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {str(e)}")
                self.redis_available = False
        
        # Fallback to in-memory storage
        return [msg.split(":", 1) for msg in self.fallback_memory]

    def clear(self):
        if self.redis_available:
            try:
                self.client.delete(self.key)
                return
            except Exception as e:
                logger.warning(f"Redis error, clearing memory fallback: {str(e)}")
                self.redis_available = False
        
        # Fallback to in-memory storage
        self.fallback_memory.clear()
