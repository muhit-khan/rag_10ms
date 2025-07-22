from typing import List, Dict, Optional
import logging
import math
import re

from utils.helpers import is_valid_text_chunk

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into chunks for vector storage"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def semantic_chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunks text by splitting it into paragraphs and then sentences,
        grouping them to respect the chunk size without needing a heavy model.
        """
        if not text:
            return []

        chunks = []
        current_chunk_text = ""
        
        # Split text into paragraphs first.
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If the paragraph is larger than the chunk size, split it into sentences.
            if len(para.split()) > self.chunk_size:
                # Add the existing chunk before processing the large paragraph
                if is_valid_text_chunk(current_chunk_text):
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "word_count": len(current_chunk_text.split()),
                        "metadata": metadata
                    })
                current_chunk_text = ""

                # Simple sentence splitting
                sentences = re.split(r'(?<=[.!?।])\s+', para)
                sentence_group = ""
                for sent in sentences:
                    if len(sentence_group.split()) + len(sent.split()) <= self.chunk_size:
                        sentence_group += " " + sent
                    else:
                        if is_valid_text_chunk(sentence_group):
                            chunks.append({
                                "text": sentence_group.strip(),
                                "word_count": len(sentence_group.split()),
                                "metadata": metadata
                            })
                        sentence_group = sent
                
                # Add the last sentence group
                if is_valid_text_chunk(sentence_group):
                    chunks.append({
                        "text": sentence_group.strip(),
                        "word_count": len(sentence_group.split()),
                        "metadata": metadata
                    })

            # If the paragraph fits, add it to the current chunk.
            elif len(current_chunk_text.split()) + len(para.split()) <= self.chunk_size:
                current_chunk_text += "\n\n" + para
            
            # Otherwise, this paragraph starts a new chunk.
            else:
                if is_valid_text_chunk(current_chunk_text):
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "word_count": len(current_chunk_text.split()),
                        "metadata": metadata
                    })
                current_chunk_text = para

        # Add the final remaining chunk
        if is_valid_text_chunk(current_chunk_text):
            chunks.append({
                "text": current_chunk_text.strip(),
                "word_count": len(current_chunk_text.split()),
                "metadata": metadata
            })
            
        return chunks

    def chunk_page(self, page_data: Dict, strategy: str = "semantic") -> List[Dict]:
        """Chunk a single page using the semantic strategy"""
        if not page_data:
            return []
        
        page_metadata = {
            "page": page_data.get("page"),
            "language": page_data.get("language"),
            "source": "HSC26_Bangla_1st_paper"
        }
        
        text = page_data.get("cleaned_text", page_data.get("text", ""))

        try:
            # Only semantic chunking is supported now.
            return self.semantic_chunk_text(text, page_metadata)
                
        except Exception as e:
            logger.error(f"Error chunking page {page_data.get('page')}: {e}")
            return []

    def chunk_document(self, preprocessed_pages: List[Dict], strategy: str = "semantic") -> List[Dict]:
        """Chunk entire document"""
        if not preprocessed_pages:
            logger.warning("No pages to chunk")
            return []
        
        logger.info(f"Chunking {len(preprocessed_pages)} pages using {strategy} strategy...")
        
        all_chunks = []
        chunk_id = 0
        
        for page_data in preprocessed_pages:
            page_chunks = self.chunk_page(page_data, strategy)
            
            # Add unique IDs to chunks
            for chunk in page_chunks:
                chunk["chunk_id"] = f"chunk_{chunk_id:04d}"
                chunk_id += 1
                all_chunks.append(chunk)
        
        # Log chunking statistics
        total_words = sum(chunk["word_count"] for chunk in all_chunks)
        avg_chunk_size = total_words / len(all_chunks) if all_chunks else 0
        
        logger.info(f"Created {len(all_chunks)} chunks, average size: {avg_chunk_size:.1f} words")
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunking process"""
        if not chunks:
            return {}
        
        word_counts = [chunk["word_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(word_counts) / len(word_counts),
            "min_chunk_size": min(word_counts),
            "max_chunk_size": max(word_counts),
            "languages": list(set(chunk["metadata"]["language"] for chunk in chunks)),
            "pages_covered": len(set(chunk["metadata"]["page"] for chunk in chunks))
        }
