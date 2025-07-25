"""
Redis client initializer
"""
import logging
from typing import Optional
import redis
from config import config

logger = logging.getLogger("redis_client")

_redis_client: Optional[redis.Redis] = None

def get_redis_client() -> redis.Redis:
    """
    Get or create a Redis client instance.
    
    Returns:
        redis.Redis: Redis client instance
        
    Raises:
        redis.ConnectionError: If unable to connect to Redis
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test the connection
            _redis_client.ping()
            logger.info(f"Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {str(e)}")
            raise redis.ConnectionError(f"Redis connection failed: {str(e)}")
    
    return _redis_client

def test_redis_connection() -> bool:
    """
    Test if Redis connection is working.
    
    Returns:
        bool: True if connection is working, False otherwise
    """
    try:
        client = get_redis_client()
        client.ping()
        return True
    except Exception as e:
        logger.warning(f"Redis connection test failed: {str(e)}")
        return False

def close_redis_connection():
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        try:
            _redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
        finally:
            _redis_client = None
