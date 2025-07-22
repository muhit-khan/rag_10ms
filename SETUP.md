# Setup Instructions

## Prerequisites

- Python 3.8+
- 4GB RAM minimum
- 2GB free disk space

## Installation Steps

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd rag_10ms

# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Minimum required: OPENAI_API_KEY (if using OpenAI)
```

### 3. Add Document

Place your HSC26 Bangla 1st paper PDF in:

```
data/raw/hsc26_bangla_1st_paper.pdf
```

### 4. First Run

```bash
# CLI Interface
python main.py

# OR API Server
python -m uvicorn api.main:app --reload
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure virtual environment is activated
2. **PDF Not Found**: Check file path in `data/raw/`
3. **Memory Issues**: Reduce chunk size in config
4. **Slow Embedding**: First run downloads models (~1GB)

### Windows Specific:

```bash
# If pip install fails
pip install --upgrade pip setuptools wheel
```

### Linux/Mac Specific:

```bash
# If PyMuPDF fails to install
apt-get install libffi-dev python3-dev
```

## Verification

```bash
# Test the installation
python -c "from src import RAGPipeline; print('✅ Installation successful')"
```
