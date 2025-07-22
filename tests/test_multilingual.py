import pytest
from unittest.mock import Mock

from utils.helpers import detect_language, clean_whitespace, normalize_query
from src.llm_client import ConversationMemory

class TestMultilingual:
    """Test multilingual capabilities"""
    
    def test_language_detection_bengali(self):
        """Test Bengali language detection"""
        bengali_texts = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "আমি একটি বাংলা বাক্য।",
            "এটি একটি পরীক্ষা।"
        ]
        
        for text in bengali_texts:
            lang = detect_language(text)
            assert lang == "bn", f"Failed to detect Bengali in: {text}"
    
    def test_language_detection_english(self):
        """Test English language detection"""
        english_texts = [
            "This is an English sentence.",
            "What is your name?",
            "Hello world!"
        ]
        
        for text in english_texts:
            lang = detect_language(text)
            assert lang == "en", f"Failed to detect English in: {text}"
    
    def test_mixed_language_detection(self):
        """Test mixed language text"""
        mixed_texts = [
            "This is অনুপম mixed text।",
            "আমি একটি English word ব্যবহার করেছি।"
        ]
        
        for text in mixed_texts:
            lang = detect_language(text)
            # Should detect the dominant language
            assert lang in ["bn", "en"]
    
    def test_whitespace_cleaning(self):
        """Test whitespace cleaning"""
        test_cases = [
            ("  multiple   spaces  ", "multiple spaces"),
            ("\n\nline\n\nbreaks\n\n", "line breaks"),
            ("\tتাب\t  spaces", "تাব spaces")
        ]
        
        for input_text, expected in test_cases:
            cleaned = clean_whitespace(input_text)
            assert cleaned == expected
    
    def test_query_normalization(self):
        """Test query normalization"""
        test_cases = [
            ("প্রশ্ন???", "প্রশ্ন"),
            ("Question!!!", "Question"),
            ("  spaced  query  ।", "spaced query"),
            ("normal query", "normal query")
        ]
        
        for input_query, expected in test_cases:
            normalized = normalize_query(input_query)
            assert normalized == expected
    
    def test_conversation_memory_multilingual(self):
        """Test conversation memory with multilingual exchanges"""
        memory = ConversationMemory(max_history=5)
        session_id = "test_session"
        
        # Add Bengali exchange
        memory.add_exchange(
            session_id, 
            "অনুপম কে?", 
            "অনুপম একটি চরিত্র।"
        )
        
        # Add English exchange
        memory.add_exchange(
            session_id,
            "Who is Anupam?",
            "Anupam is a character."
        )
        
        history = memory.get_history(session_id)
        
        assert len(history) == 2
        assert history[0]["user_query"] == "অনুপম কে?"
        assert history[1]["user_query"] == "Who is Anupam?"
    
    def test_recent_queries_extraction(self):
        """Test extraction of recent queries for context"""
        memory = ConversationMemory(max_history=10)
        session_id = "test_session"
        
        queries = [
            "প্রথম প্রশ্ন",
            "দ্বিতীয় প্রশ্ন", 
            "Third question",
            "চতুর্থ প্রশ্ন"
        ]
        
        for query in queries:
            memory.add_exchange(session_id, query, f"উত্তর: {query}")
        
        recent = memory.get_recent_queries(session_id, count=2)
        
        assert len(recent) == 2
        assert recent == ["Third question", "চতুর্থ প্রশ্ন"]
    
    def test_memory_max_history_limit(self):
        """Test memory history limit"""
        memory = ConversationMemory(max_history=3)
        session_id = "test_session"
        
        # Add more exchanges than the limit
        for i in range(5):
            memory.add_exchange(
                session_id,
                f"প্রশ্ন {i+1}",
                f"উত্তর {i+1}"
            )
        
        history = memory.get_history(session_id)
        
        assert len(history) == 3  # Should keep only last 3
        assert history[0]["user_query"] == "প্রশ্ন ৩"  # Should have latest ones
        assert history[-1]["user_query"] == "প্রশ্ন ৫"
    
    def test_session_isolation(self):
        """Test that different sessions are isolated"""
        memory = ConversationMemory()
        
        memory.add_exchange("session1", "প্রশ্ন ১", "উত্তর ১")
        memory.add_exchange("session2", "Question 1", "Answer 1")
        
        history1 = memory.get_history("session1")
        history2 = memory.get_history("session2")
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["user_query"] == "প্রশ্ন ১"
        assert history2[0]["user_query"] == "Question 1"
    
    def test_empty_session_handling(self):
        """Test handling of non-existent sessions"""
        memory = ConversationMemory()
        
        history = memory.get_history("non_existent")
        recent = memory.get_recent_queries("non_existent")
        
        assert history == []
        assert recent == []
    
    def test_session_clearing(self):
        """Test session clearing functionality"""
        memory = ConversationMemory()
        session_id = "test_session"
        
        memory.add_exchange(session_id, "প্রশ্ন", "উত্তর")
        assert len(memory.get_history(session_id)) == 1
        
        memory.clear_session(session_id)
        assert len(memory.get_history(session_id)) == 0
