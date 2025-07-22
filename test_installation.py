"""Simple installation test"""

def test_imports():
    """Test if all critical packages can be imported"""
    
    try:
        # Core packages
        import fastapi
        print("✅ FastAPI imported successfully")
        
        import pydantic
        print("✅ Pydantic imported successfully")
        
        import uvicorn
        print("✅ Uvicorn imported successfully")
        
        # PDF processing
        import pdfplumber
        print("✅ PDFplumber imported successfully")
        
        # NLP packages
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
        
        # Vector database
        import chromadb
        print("✅ ChromaDB imported successfully")
        
        # Scientific computing
        import numpy
        print("✅ NumPy imported successfully")
        
        import pandas
        print("✅ Pandas imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        # Utilities
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
        
        print("\n🎉 All critical packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
