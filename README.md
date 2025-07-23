# 🚀 Resume Parser with Advanced NLP & Prolog Screening

An intelligent resume screening system that combines advanced Natural Language Processing (NLP) with Prolog rule-based reasoning to analyze and evaluate resumes with high accuracy.

## ✨ Features

### 🧠 Advanced NLP Processing
- **Real Document Support**: Extract text from PDF, DOCX, and TXT files using proper libraries
- **spaCy Integration**: Advanced named entity recognition for contact information
- **Contextual Analysis**: Intelligent skill extraction using semantic similarity
- **Pattern Matching**: Advanced regex and spaCy patterns for robust data extraction

### 🔍 Prolog Rule Engine
- **Logic-Based Screening**: Rules written in Prolog-style syntax for transparent decision making
- **Configurable Criteria**: Easily modify screening rules for different job requirements
- **Confidence Scoring**: Each decision comes with a confidence score (60-95%)
- **Detailed Explanations**: Clear breakdown of why each decision was made

### 🎨 Beautiful Modern UI
- **Glassmorphism Design**: Modern, elegant interface with gradient backgrounds
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Real-time Results**: Instant analysis and beautiful result visualization

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resume-parser.git
cd resume-parser
```

2. **Create and activate virtual environment**
```bash
# Using conda
conda create -n resume-parser python=3.9
conda activate resume-parser

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies and setup NLP**
```bash
python setup_nlp.py
```

4. **Run the application**
```bash
python prolog_resume_app.py
```

5. **Open your browser** and navigate to `http://localhost:5002`

## 📋 Requirements

The system automatically installs the following dependencies:

### Core Framework
- `Flask==2.3.3` - Web framework
- `Werkzeug==2.3.7` - WSGI utilities

### NLP & Document Processing
- `spacy==3.7.2` - Advanced NLP processing
- `PyPDF2==3.0.1` - PDF text extraction
- `python-docx==1.1.0` - DOCX text extraction

### Additional Models
- `en_core_web_sm` - spaCy English model (automatically downloaded)

## 🎯 How It Works

### 1. Document Processing
```python
# Enhanced text extraction
if file.endswith('.pdf'):
    text = extract_with_PyPDF2(file)
elif file.endswith('.docx'):
    text = extract_with_python_docx(file)
```

### 2. NLP Analysis
```python
# Advanced entity recognition
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Extract entities with context
for ent in doc.ents:
    if ent.label_ == "PERSON":
        candidates.append(ent.text)
```

### 3. Prolog Rule Evaluation
```python
# Rule-based screening
if evaluate("qualified_senior_developer(X)", candidate):
    decision = "HIRE"
    confidence = 0.95
    recommendation = "Highly qualified senior developer"
```

## 🔧 Configuration

### Adding New Skills
Edit the `skills_keywords` dictionary in `AdvancedResumeParser`:

```python
self.skills_keywords = {
    'programming': ['python', 'java', 'javascript', 'rust', 'go'],
    'ml_ai': ['tensorflow', 'pytorch', 'scikit-learn'],
    # Add your categories here
}
```

### Modifying Screening Rules
Update the rules in the `PrologEngine` class:

```python
self.rules = [
    ("senior_level(X)", lambda x: x.get('years_experience', 0) >= 5),
    ("has_leadership_skills(X)", lambda x: 'management' in x.get('skills', [])),
    # Add your custom rules here
]
```

## 📊 Screening Criteria

### Current Evaluation Rules

1. **Education Level**
   - ✅ High Education: Bachelor's, Master's, PhD detection
   - 🎓 University Recognition: NLP-based institution extraction

2. **Experience Assessment**
   - ⏰ Experience Calculation: Multiple date range parsing methods
   - 📈 Senior Level: 5+ years experience threshold
   - 💼 Career Progression: Role and responsibility analysis

3. **Technical Skills**
   - 💻 Programming Languages: 13+ languages supported
   - 🌐 Web Technologies: Frontend/backend framework detection
   - 🗄️ Database Systems: SQL and NoSQL database recognition
   - ☁️ Cloud & DevOps: Modern infrastructure tools

4. **Decision Matrix**
   - **HIRE (95% confidence)**: Senior developer with comprehensive skills
   - **HIRE (85% confidence)**: Strong web developer candidate
   - **CONSIDER (75% confidence)**: Qualified developer with potential
   - **REJECT (60% confidence)**: Below minimum requirements

## 🎨 UI Components

### Upload Interface
- Drag & drop file upload
- File type validation
- Progress indicators
- Error handling with user-friendly messages

### Results Dashboard
- Candidate overview cards
- Detailed skill breakdowns
- Rule evaluation matrices
- Confidence score visualizations

## 🔍 API Endpoints

### Core Routes
- `GET /` - Main landing page
- `GET /upload` - File upload interface
- `POST /upload` - Process resume upload
- `GET /results` - View all screening results
- `GET /results/<id>` - View specific candidate results

### API Routes
- `GET /api/candidates` - JSON data for all candidates

## 🧪 Testing

### Manual Testing
1. Upload various resume formats (PDF, DOCX, TXT)
2. Test with different experience levels
3. Verify skill extraction accuracy
4. Check decision logic consistency

### Example Test Cases
- Senior developer with 7+ years (Expected: HIRE 95%)
- Recent graduate with strong skills (Expected: CONSIDER 75%)
- Career changer with minimal experience (Expected: REJECT 60%)

## 🚀 Deployment

### Local Development
```bash
python prolog_resume_app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5002 prolog_resume_app:app
```

## 📈 Performance

### NLP Processing
- ⚡ Fast text extraction from documents
- 🧠 Contextual skill analysis using spaCy embeddings
- 📝 Accurate named entity recognition

### Scalability
- 💾 In-memory storage for demo (easily replaceable with database)
- 🔄 Stateless design for horizontal scaling
- 📊 Efficient rule evaluation engine

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** for advanced NLP capabilities
- **Flask** for the elegant web framework
- **PyPDF2** and **python-docx** for document processing
- **Modern UI Design** inspiration from glassmorphism trends

## 📞 Support

For support, email niloyrahman1337@gmail.com or create an issue in this repository.

---

**Made with ❤️ and advanced NLP technology** 
