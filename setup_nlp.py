#!/usr/bin/env python3
"""
Setup script for Resume Parser with Advanced NLP
Installs required packages and downloads spaCy model
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Resume Parser with Advanced NLP")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment. Consider using conda or venv.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled. Please activate a virtual environment first.")
            return
    
    # Install Python packages
    print("\n📦 Installing Python packages...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        return
    
    # Download spaCy model
    print("\n🧠 Downloading spaCy English model...")
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("❌ Failed to download spaCy model. Please check your internet connection.")
        print("You can try manually: python -m spacy download en_core_web_sm")
        return
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import spacy
        import PyPDF2
        from docx import Document
        nlp = spacy.load("en_core_web_sm")
        print("✅ All packages installed successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run the resume parser with:")
    print("python prolog_resume_app.py")
    print("\nThe enhanced parser now supports:")
    print("• Real PDF and DOCX text extraction")
    print("• Advanced NLP-based information extraction")
    print("• Better name, email, and phone detection")
    print("• Contextual skill extraction")
    print("• Improved education and experience parsing")

if __name__ == "__main__":
    main() 