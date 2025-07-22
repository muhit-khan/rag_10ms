# Installation Guide for Windows

## Problem: PyMuPDF Installation Failure

The error you encountered is because **PyMuPDF requires Microsoft Visual Studio Build Tools** to compile on Windows, which you don't have installed.

## Quick Solution

Use the Windows-specific requirements file I created:

```bash
# Instead of the original requirements.txt, use:
pip install -r requirements-windows.txt
```

This version:

- ✅ **Removes PyMuPDF** (which needs Visual Studio)
- ✅ **Uses pdfplumber** as primary PDF processor
- ✅ **Keeps all other dependencies** intact
- ✅ **Maintains full Bengali text support**

## Alternative Solutions

### Option 1: Install Pre-compiled PyMuPDF (Recommended)

```bash
# Try installing a pre-compiled wheel
pip install --only-binary=PyMuPDF PyMuPDF
```

### Option 2: Install Visual Studio Build Tools

1. Download **Microsoft C++ Build Tools**
2. Install with **C++ CMake tools** and **Windows SDK**
3. Restart terminal and try original requirements.txt

### Option 3: Use Conda (Alternative)

```bash
# If you prefer conda
conda install -c conda-forge pymupdf
```

## Verify Installation

After installing with `requirements-windows.txt`:

```bash
# Test the installation
python -c "from src import RAGPipeline; print('✅ Installation successful')"
```

## System Impact

**No functionality loss**: The system will work perfectly with pdfplumber for PDF text extraction. For Bengali text processing, pdfplumber is actually quite good and handles UTF-8 properly.

**Performance**: Slightly different text extraction, but comparable quality for your use case.

## Next Steps

1. Install using: `pip install -r requirements-windows.txt`
2. Continue with setup as normal
3. Place your PDF in `data/raw/`
4. Run `python main.py`

The system will automatically detect available PDF processors and use the best one available.
