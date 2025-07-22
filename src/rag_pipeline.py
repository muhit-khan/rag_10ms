from typing import Dict, List, Optional
import logging
import asyncio
import os

from src.pdf_processor import PDFProcessor
from src.text_cleaner import BengaliTextCleaner
from src.chunking import DocumentChunker
from src.embeddings import EmbeddingModel, VectorStore
from src.retrieval import Retriever
from src.llm_client import LLMClient, ConversationMemory
from config import settings
from utils import setup_logging, detect_language, ensure_directory_exists

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self):
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = BengaliTextCleaner()
        self.chunker = DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
        self.vector_store = VectorStore(
            embedding_model=self.embedding_model,
            save_path=os.path.join(settings.PROCESSED_DATA_DIR, "vector_store.pkl")
        )
        self.retriever = Retriever(self.vector_store)
        self.llm_client = LLMClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL
        )
        self.memory = ConversationMemory(max_history=settings.MAX_CHAT_HISTORY)
        self.is_initialized = False
        self.knowledge_base_loaded = False # Explicitly define the attribute

    async def initialize(self, pdf_path: Optional[str] = None):
        """
        Initializes the RAG pipeline by loading the vector store or building it from a PDF.
        """
        logger.info("Initializing RAG pipeline...")
        
        # Try to load an existing vector store first
        if self.vector_store.exists():
            logger.info("Loading existing vector store...")
            self.vector_store.load()
            logger.info("Vector store loaded successfully.")
            self.is_initialized = True
            self.knowledge_base_loaded = self.vector_store.is_loaded()
        elif pdf_path and os.path.exists(pdf_path):
            logger.info(f"No existing vector store found. Processing PDF: {pdf_path}")
            await self.load_and_process_pdf(pdf_path)
        else:
            logger.warning("Vector store not found and no PDF provided. System is not initialized.")
            self.is_initialized = False

        if self.is_initialized:
            logger.info("RAG pipeline initialized successfully.")
        else:
            logger.warning("RAG pipeline initialization failed or was incomplete.")

    def is_ready(self) -> bool:
        """Check if the pipeline is initialized and ready to process queries."""
        return self.is_initialized and self.vector_store is not None and self.vector_store.is_loaded()

    async def load_and_process_pdf(self, pdf_path: str):
        """Loads a PDF, processes it, and builds the vector store."""
        try:
            logger.info(f"Starting PDF processing for: {pdf_path}")
            
            # 1. Process PDF and save extracted text
            pages = await self.pdf_processor.process_pdf(pdf_path)
            
            if not pages:
                logger.error("No pages extracted from PDF")
                self.is_initialized = False
                return
            
            # Save the enhanced extracted text for inspection
            extracted_text_path = os.path.join(settings.PROCESSED_DATA_DIR, "enhanced_extracted_text.txt")
            self.pdf_processor.save_extracted_text(pages, extracted_text_path)
            logger.info(f"Enhanced extracted text saved to: {extracted_text_path}")
            
            # Also save extraction statistics
            stats = self.pdf_processor.get_extraction_stats(pages)
            logger.info(f"Extraction stats - Pages: {stats['total_pages']}, "
                       f"Avg Quality: {stats['avg_quality_score']:.1f}, "
                       f"Bengali Ratio: {stats['avg_bengali_ratio']:.1%}")
            
            # 2. Clean and preprocess the pages
            preprocessed_pages = self.text_cleaner.preprocess_document(pages)
            
            if not preprocessed_pages:
                logger.error("No valid pages after text cleaning")
                self.is_initialized = False
                return
            
            # 3. Chunk Document
            chunks = self.chunker.chunk_document(preprocessed_pages)
            
            if not chunks:
                logger.error("No chunks were created from the PDF. Aborting initialization.")
                self.is_initialized = False
                return

            # 4. Build Vector Store
            logger.info(f"Building vector store from {len(chunks)} chunks...")
            self.vector_store.build(chunks)
            self.vector_store.save() # Save the newly built store
            
            self.is_initialized = True
            self.knowledge_base_loaded = True
            logger.info("PDF processed and vector store built successfully.")

        except Exception as e:
            logger.error(f"An error occurred during PDF processing and vector store creation: {e}", exc_info=True)
            self.is_initialized = False

    async def process_document(self, pdf_path: str) -> bool:
        """Process a PDF document into the knowledge base"""
        try:
            logger.info(f"Starting document processing: {pdf_path}")
            
            # Step 1: Extract text from PDF
            pages_data = self.pdf_processor.extract_text(pdf_path)
            
            if not pages_data:
                logger.error("No text extracted from PDF")
                return False
            
            # Save extracted text for inspection (enhanced version)
            extracted_text_path = os.path.join(settings.PROCESSED_DATA_DIR, "enhanced_extracted_text.txt")
            self.pdf_processor.save_extracted_text(pages_data, extracted_text_path)
            
            # Log extraction statistics
            if hasattr(self.pdf_processor, 'get_extraction_stats'):
                stats = self.pdf_processor.get_extraction_stats(pages_data)
                logger.info(f"Extraction Quality - Pages: {stats.get('total_pages', 0)}, "
                           f"Avg Score: {stats.get('avg_quality_score', 0):.1f}, "
                           f"Bengali Ratio: {stats.get('avg_bengali_ratio', 0):.1%}")
            else:
                logger.info(f"Extracted {len(pages_data)} pages from PDF")
            
            # Step 2: Clean and preprocess text
            preprocessed_pages = self.text_cleaner.preprocess_document(pages_data)
            
            if not preprocessed_pages:
                logger.error("No valid pages after preprocessing")
                return False
            
            # Step 3: Chunk the document
            chunks = self.chunker.chunk_document(preprocessed_pages, strategy="sentences")
            
            if not chunks:
                logger.error("No chunks created")
                return False
            
            # Step 4: Create embeddings and store
            if not self.vector_store.add_chunks(chunks):
                logger.error("Failed to add chunks to vector store")
                return False
            
            # Step 5: Save vector store
            vector_store_path = os.path.join(settings.PROCESSED_DATA_DIR, "vector_store.pkl")
            
            if self.vector_store.save(vector_store_path):
                self.knowledge_base_loaded = True
                
                # Log processing statistics
                stats = self.vector_store.get_stats()
                chunking_stats = self.chunker.get_chunking_stats(chunks)
                
                logger.info("Document processing completed:")
                logger.info(f"- Total chunks: {stats['total_chunks']}")
                logger.info(f"- Languages: {stats['languages']}")
                logger.info(f"- Average chunk size: {chunking_stats['avg_chunk_size']:.1f} words")
                
                return True
            else:
                logger.error("Failed to save vector store")
                return False
                
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return False
    
    async def process_query(self, 
                          query: str, 
                          session_id: str = "default",
                          language: Optional[str] = None) -> Dict:
        """Process a user query and generate response"""
        
        if not self.is_initialized or self.retriever is None:
            return {
                "answer": "System not initialized. Please load a document first.",
                "sources": [],
                "session_id": session_id,
                "language_detected": "en"
            }
        
        try:
            # Detect language if not provided
            if not language:
                language = detect_language(query)
            
            logger.info(f"Processing query (lang: {language}): {query[:100]}...")
            
            # Get conversation history
            conversation_history = self.memory.get_history(session_id)
            recent_queries = self.memory.get_recent_queries(session_id)
            
            # Retrieve relevant documents
            retrieval_result = self.retriever.retrieve_with_context(
                query=query,
                conversation_history=recent_queries,
                k=settings.TOP_K_RETRIEVAL
            )

            # If no relevant context is found, return a specific message
            if not retrieval_result or not retrieval_result["sources"]:
                logger.warning(f"No relevant documents found for query: {query}")
                answer = "দুঃখিত, এই বিষয়ে আমার কাছে কোনো তথ্য নেই।" if language == "bn" else "Sorry, I don't have any information on this topic."
                self.memory.add_exchange(session_id, query, answer)
                return {
                    "answer": answer,
                    "sources": [],
                    "session_id": session_id,
                    "language_detected": language,
                }
            
            # Generate response using LLM
            response = self.llm_client.generate_response(
                query=query,
                context=retrieval_result["context"],
                conversation_history=conversation_history,
                language=language
            )
            
            # Store conversation
            self.memory.add_exchange(session_id, query, response)
            
            # Prepare response
            result = {
                "answer": response,
                "sources": retrieval_result["sources"],
                "session_id": session_id,
                "language_detected": language,
                "retrieval_stats": {
                    "chunks_found": retrieval_result["total_chunks"],
                    "avg_score": retrieval_result["avg_score"]
                }
            }
            
            logger.info(f"Query processed successfully. Response length: {len(response)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "দুঃখিত, আপনার প্রশ্নের উত্তর দিতে সমস্যা হয়েছে।" if language == "bn" else "Sorry, there was an error processing your question.",
                "sources": [],
                "session_id": session_id,
                "language_detected": language or "bn",
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            "pipeline_status": "initialized" if self.is_initialized else "not_initialized",
            "knowledge_base_loaded": self.knowledge_base_loaded
        }
        
        if self.knowledge_base_loaded:
            stats.update(self.vector_store.get_stats())
        
        if self.retriever:
            stats["retrieval_stats"] = self.retriever.get_retrieval_stats()
        
        # Memory stats
        total_sessions = len(self.memory.conversations)
        total_exchanges = sum(len(conv) for conv in self.memory.conversations.values())
        
        stats.update({
            "active_sessions": total_sessions,
            "total_conversations": total_exchanges
        })
        
        return stats
    
    async def add_document(self, pdf_path: str) -> bool:
        """Add additional document to the knowledge base"""
        return await self.process_document(pdf_path)
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear_session(session_id)
