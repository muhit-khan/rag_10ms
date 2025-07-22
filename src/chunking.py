from typing import List, Dict, Optional
import logging
import math

from utils.helpers import is_valid_text_chunk

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into chunks for vector storage"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_by_sentences(self, sentences: List[str], metadata: Dict) -> List[Dict]:
        """Chunk text by grouping sentences"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Calculate potential chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            potential_word_count = len(potential_chunk.split())
            
            if potential_word_count <= self.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it's valid
                if current_chunk and is_valid_text_chunk(current_chunk):
                    chunks.append({
                        "text": current_chunk.strip(),
                        "sentences": current_sentences.copy(),
                        "word_count": len(current_chunk.split()),
                        "metadata": metadata
                    })
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_sentences:
                    overlap_sentences = current_sentences[-self.overlap:]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
        
        # Add final chunk
        if current_chunk and is_valid_text_chunk(current_chunk):
            chunks.append({
                "text": current_chunk.strip(),
                "sentences": current_sentences,
                "word_count": len(current_chunk.split()),
                "metadata": metadata
            })
        
        return chunks
    
    def chunk_by_paragraphs(self, paragraphs: List[str], metadata: Dict) -> List[Dict]:
        """Chunk text by paragraphs with smart splitting"""
        if not paragraphs:
            return []
        
        chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_word_count = len(paragraph.split())
            
            if para_word_count <= self.chunk_size:
                # Paragraph fits in one chunk
                chunks.append({
                    "text": paragraph,
                    "word_count": para_word_count,
                    "paragraph_index": para_idx,
                    "metadata": metadata
                })
            else:
                # Split large paragraph into smaller chunks
                words = paragraph.split()
                
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if is_valid_text_chunk(chunk_text):
                        chunks.append({
                            "text": chunk_text,
                            "word_count": len(chunk_words),
                            "paragraph_index": para_idx,
                            "chunk_part": i // (self.chunk_size - self.overlap),
                            "metadata": metadata
                        })
        
        return chunks
    
    def chunk_by_sliding_window(self, text: str, metadata: Dict) -> List[Dict]:
        """Create overlapping chunks using sliding window"""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= self.chunk_size:
            return [{
                "text": text,
                "word_count": len(words),
                "metadata": metadata
            }]
        
        chunks = []
        step_size = self.chunk_size - self.overlap
        
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if is_valid_text_chunk(chunk_text):
                chunks.append({
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "chunk_index": i // step_size,
                    "metadata": metadata
                })
        
        return chunks
    
    def chunk_page(self, page_data: Dict, strategy: str = "sentences") -> List[Dict]:
        """Chunk a single page using specified strategy"""
        if not page_data:
            return []
        
        page_metadata = {
            "page": page_data.get("page"),
            "language": page_data.get("language"),
            "source": "HSC26_Bangla_1st_paper"
        }
        
        try:
            if strategy == "sentences" and page_data.get("sentences"):
                return self.chunk_by_sentences(page_data["sentences"], page_metadata)
                
            elif strategy == "paragraphs" and page_data.get("paragraphs"):
                return self.chunk_by_paragraphs(page_data["paragraphs"], page_metadata)
                
            elif strategy == "sliding_window":
                text = page_data.get("cleaned_text", page_data.get("text", ""))
                return self.chunk_by_sliding_window(text, page_metadata)
            
            else:
                # Fallback to sliding window on cleaned text
                text = page_data.get("cleaned_text", page_data.get("text", ""))
                return self.chunk_by_sliding_window(text, page_metadata)
                
        except Exception as e:
            logger.error(f"Error chunking page {page_data.get('page')}: {e}")
            return []
    
    def chunk_document(self, preprocessed_pages: List[Dict], strategy: str = "sentences") -> List[Dict]:
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
