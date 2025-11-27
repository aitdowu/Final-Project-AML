#!/bin/bash
# Reset Vector Database Script
# This script deletes the old vector database and regenerates it with the current embedding model

cd "$(dirname "$0")"

echo "=========================================="
echo "Resetting Vector Database"
echo "=========================================="
echo ""

# Check if db directory exists
if [ -d "db" ]; then
    echo "Step 1: Deleting old database..."
    rm -rf db/
    echo "✓ Old database deleted"
else
    echo "Step 1: No existing database found (this is OK)"
fi

echo ""
echo "Step 2: Regenerating database with current embedding model..."
echo "  (This may take a few minutes depending on the number of PDFs)"
echo ""

# Run ingestion script
python Pipeline/main_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Database reset complete!"
    echo "=========================================="
    echo ""
    echo "You can now run the chatbot:"
    echo "  python Pipeline/rag_chatbot.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Error during database regeneration"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "Make sure:"
    echo "  1. Your PDFs are in the data/ directory"
    echo "  2. The embedding model can be loaded"
    echo "  3. All dependencies are installed"
    echo ""
    exit 1
fi

