"""
Conversation buffer logic
"""
from db.redis_client import get_redis_client
from config import config

class RedisWindow:
    def __init__(self, user_id: str, window_size: int = 4):
        self.client = get_redis_client()
        self.user_id = user_id
        self.window_size = window_size
        self.key = f"chat:{user_id}"

    def add_message(self, role: str, content: str):
        self.client.rpush(self.key, f"{role}:{content}")
        self.client.ltrim(self.key, -self.window_size, -1)

    def get_window(self):
        messages = self.client.lrange(self.key, 0, -1)
        return [msg.split(":", 1) for msg in messages]

    def clear(self):
        self.client.delete(self.key)
