#!/bin/bash
# BMAD Music Production Expansion Pack Initialization
# Provides the /init command functionality referenced in CLAUDE.md

echo
echo "🎵 BMAD Music Production System Initialization"
echo "==============================================="
echo

# Run the BMAD initialization script
python3 bmad_init.py "$@"

# Check if Python script was successful
if [ $? -eq 0 ]; then
    echo
    echo "✅ BMAD initialization completed successfully!"
    echo "🎯 Ready for hardcore music production with specialized agents!"
    echo
else
    echo
    echo "❌ BMAD initialization failed!"
    echo "💡 Check the error messages above and try again."
    echo
    exit 1
fi