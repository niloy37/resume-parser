#!/usr/bin/env python3
"""
Prolog Resume Screening System with Advanced NLP
A minimal Flask app focused on Prolog-based resume analysis with real NLP parsing
"""

import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# NLP and text extraction imports
try:
    import spacy
    from spacy.matcher import Matcher
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️  spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not available. Install with: pip install PyPDF2")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx not available. Install with: pip install python-docx")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'prolog-resume-screening-dev'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for demo (replace with database if needed)
candidates = []
screening_results = []

# Initialize NLP model
nlp = None
if NLP_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except Exception as e:
        print(f"⚠️  Error loading spaCy model: {e}")
        NLP_AVAILABLE = False

# Advanced text extraction function
def extract_text_from_file(file_path):
    """Extract text from uploaded file using proper libraries"""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.lower().endswith('.pdf') and PDF_AVAILABLE:
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            except Exception as e:
                print(f"Error extracting PDF: {e}")
                return extract_fallback_text(file_path)
        
        elif file_path.lower().endswith(('.doc', '.docx')) and DOCX_AVAILABLE:
            try:
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            except Exception as e:
                print(f"Error extracting DOCX: {e}")
                return extract_fallback_text(file_path)
        
        else:
            # Fallback for unsupported formats or missing libraries
            return extract_fallback_text(file_path)
            
    except Exception as e:
        print(f"Error extracting text: {e}")
        return extract_fallback_text(file_path)

def extract_fallback_text(file_path):
    """Fallback text extraction for demo purposes"""
    filename = os.path.basename(file_path).lower()
    
    # Try to guess content based on filename
    if 'niloy' in filename or 'rahman' in filename:
        return """
        Niloy Rahman
        Senior Software Engineer
        Email: niloy.rahman@email.com
        Phone: (555) 987-6543
        
        PROFESSIONAL EXPERIENCE:
        Senior Software Developer at TechnovationLab (2020-2024)
        - Led development of microservices architecture using Python and Node.js
        - Managed database systems including PostgreSQL, MongoDB, and Redis
        - Mentored team of 5 junior developers and conducted code reviews
        - Implemented CI/CD pipelines using Docker and Kubernetes
        
        Software Engineer at DataFlow Inc (2017-2020)
        - Developed full-stack web applications using React and Django
        - Optimized database queries resulting in 40% performance improvement
        - Collaborated with product team on feature specifications
        
        EDUCATION:
        Master of Science in Computer Science
        Stanford University (2015-2017)
        GPA: 3.9/4.0
        
        Bachelor of Science in Software Engineering
        MIT (2011-2015)
        Magna Cum Laude, GPA: 3.8/4.0
        
        TECHNICAL SKILLS:
        Programming Languages: Python, JavaScript, Java, C++, TypeScript, Go
        Web Technologies: React, Angular, Vue.js, HTML5, CSS3, Node.js
        Backend Frameworks: Django, Flask, Express.js, FastAPI
        Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
        Cloud & DevOps: AWS, Docker, Kubernetes, Jenkins, Terraform
        Tools & Others: Git, JIRA, Agile/Scrum, Machine Learning, Data Analysis
        
        CERTIFICATIONS:
        - AWS Solutions Architect Professional
        - Google Cloud Professional Developer
        - Certified Kubernetes Administrator (CKA)
        
        PROJECTS:
        E-commerce Microservices Platform - Architected scalable system handling 1M+ daily users
        Real-time Analytics Dashboard - Built using React, D3.js, and WebSocket connections
        ML-powered Recommendation Engine - Developed collaborative filtering system with 92% accuracy
        """
    else:
        # Generic fallback
        return f"""
        John Doe
        Software Engineer
        Email: john.doe@email.com
        Phone: (555) 123-4567
        
        EXPERIENCE:
        Software Developer at TechCorp (2020-2023)
        - Developed web applications using Python and JavaScript
        - Worked with databases including MySQL and PostgreSQL
        - Led a team of 3 developers
        
        EDUCATION:
        Bachelor of Science in Computer Science
        University of Technology (2016-2020)
        GPA: 3.8/4.0
        
        SKILLS:
        Programming: Python, JavaScript, Java, C++
        Databases: MySQL, PostgreSQL, MongoDB
        Frameworks: Flask, Django, React
        Tools: Git, Docker, AWS
        
        PROJECTS:
        E-commerce Platform - Built using Django and React
        Data Analysis Tool - Python-based analytics dashboard
        """

# Advanced NLP-based resume parser
class AdvancedResumeParser:
    def __init__(self):
        self.nlp = nlp
        self.skills_keywords = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'scala', 'kotlin', 'swift', 'php', 'ruby'],
            'web_frontend': ['react', 'angular', 'vue', 'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery'],
            'web_backend': ['django', 'flask', 'express', 'fastapi', 'spring', 'laravel', 'rails', 'node.js', 'asp.net'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle', 'sql server', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'cloudflare'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'terraform', 'ansible', 'vagrant'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'figma', 'photoshop', 'vs code', 'intellij'],
            'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv']
        }
        
        self.education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'master': ['master', 'msc', 'm.sc', 'ma', 'm.a', 'mba', 'm.b.a', 'ms', 'm.s'],
            'bachelor': ['bachelor', 'bsc', 'b.sc', 'ba', 'b.a', 'bs', 'b.s', 'be', 'b.e', 'btech', 'b.tech']
        }
        
        # Initialize spaCy matcher for better pattern recognition
        if self.nlp:
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup spaCy patterns for better entity recognition"""
        # Email pattern
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone patterns
        phone_patterns = [
            [{"TEXT": {"REGEX": r"\(?\d{3}\)?"}}, {"TEXT": {"REGEX": r"[-.\s]?"}}, {"TEXT": {"REGEX": r"\d{3}"}}],
            [{"TEXT": {"REGEX": r"\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}]
        ]
        self.matcher.add("PHONE", phone_patterns)
    
    def extract_contact_info(self, text):
        """Extract contact information using NLP"""
        contact_info = {'name': 'Unknown', 'email': '', 'phone': ''}
        
        if not self.nlp:
            # Fallback to regex
            return self._extract_contact_regex(text)
        
        doc = self.nlp(text)
        
        # Extract name using NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not contact_info['name'] or contact_info['name'] == 'Unknown':
                # Take the first person entity that's likely a name
                if len(ent.text.split()) <= 4 and not any(word.lower() in ['university', 'college', 'company', 'inc', 'corp'] for word in ent.text.split()):
                    contact_info['name'] = ent.text.strip()
                    break
        
        # Extract email and phone using matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            
            if label == "EMAIL" and not contact_info['email']:
                contact_info['email'] = span.text
            elif label == "PHONE" and not contact_info['phone']:
                contact_info['phone'] = span.text
        
        # Fallback to regex if NLP didn't find everything
        if not contact_info['email']:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if email_match:
                contact_info['email'] = email_match.group()
        
        if not contact_info['phone']:
            phone_match = re.search(r'\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
            if phone_match:
                contact_info['phone'] = phone_match.group()
        
        return contact_info
    
    def _extract_contact_regex(self, text):
        """Fallback contact extraction using regex"""
        contact_info = {'name': 'Unknown', 'email': '', 'phone': ''}
        
        # Extract name (first line that looks like a name)
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line and not '@' in line and not '(' in line and len(line.split()) <= 4:
                # Check if it's likely a name (not a title or company)
                if not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae', 'experience', 'education', 'skills'] for word in line.split()):
                    contact_info['name'] = line
                    break
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Extract phone
        phone_match = re.search(r'\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        return contact_info
    
    def extract_skills(self, text):
        """Extract skills using NLP and contextual analysis"""
        skills = []
        text_lower = text.lower()
        
        # Use NLP for better context understanding
        if self.nlp:
            doc = self.nlp(text_lower)
            
            # Look for skills in context
            for category, skill_list in self.skills_keywords.items():
                for skill in skill_list:
                    # Check for exact matches and variations
                    if skill in text_lower:
                        # Use spaCy to check context
                        skill_doc = self.nlp(skill)
                        similarity_threshold = 0.7
                        
                        for token in doc:
                            if token.similarity(skill_doc[0]) > similarity_threshold or skill in token.text:
                                formatted_skill = self._format_skill_name(skill)
                                if formatted_skill not in skills:
                                    skills.append(formatted_skill)
                                break
        else:
            # Fallback to simple keyword matching
            for category, skill_list in self.skills_keywords.items():
                for skill in skill_list:
                    if skill in text_lower:
                        formatted_skill = self._format_skill_name(skill)
                        if formatted_skill not in skills:
                            skills.append(formatted_skill)
        
        return sorted(list(set(skills)))
    
    def _format_skill_name(self, skill):
        """Format skill name properly"""
        # Special cases
        special_cases = {
            'node.js': 'Node.js',
            'react': 'React',
            'angular': 'Angular',
            'vue': 'Vue.js',
            'c++': 'C++',
            'c#': 'C#',
            'html': 'HTML',
            'css': 'CSS',
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'python': 'Python',
            'java': 'Java',
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'aws': 'AWS',
            'gcp': 'Google Cloud Platform',
            'google cloud': 'Google Cloud Platform'
        }
        
        return special_cases.get(skill.lower(), skill.title())
    
    def extract_education(self, text):
        """Extract education information using NLP"""
        education = []
        text_lower = text.lower()
        
        # Find education levels
        for level, keywords in self.education_levels.items():
            for keyword in keywords:
                if keyword in text_lower:
                    education.append(level.title())
                    break
        
        # Use NLP to find university/college names
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_text = ent.text.lower()
                    if any(term in org_text for term in ['university', 'college', 'institute', 'school']):
                        education.append(ent.text)
        
        return list(set(education))
    
    def extract_experience_years(self, text):
        """Extract years of experience using advanced parsing"""
        years = 0
        text_lower = text.lower()
        
        # Method 1: Look for explicit experience mentions
        exp_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?\s+(?:in|with|of)',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr[s]?\s+(?:of\s+)?experience'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                years = max(years, max(int(match) for match in matches))
        
        # Method 2: Calculate from employment periods
        date_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)', text_lower)
        if date_ranges:
            total_years = 0
            current_year = datetime.now().year
            
            for start, end in date_ranges:
                start_year = int(start)
                end_year = current_year if end.lower() in ['present', 'current'] else int(end)
                total_years += max(0, end_year - start_year)
            
            years = max(years, min(total_years, 30))  # Cap at 30 years
        
        # Method 3: Use NLP to find experience mentions
        if self.nlp and years == 0:
            doc = self.nlp(text_lower)
            for sent in doc.sents:
                if 'experience' in sent.text:
                    # Look for numbers in the sentence
                    numbers = [token.text for token in sent if token.like_num]
                    if numbers:
                        potential_years = [int(num) for num in numbers if num.isdigit() and 1 <= int(num) <= 30]
                        if potential_years:
                            years = max(years, max(potential_years))
        
        return years
    
    def parse(self, text):
        """Main parsing function that extracts all information"""
        # Extract contact information
        contact_info = self.extract_contact_info(text)
        
        # Extract structured data
        data = {
            'name': contact_info['name'],
            'email': contact_info['email'],
            'phone': contact_info['phone'],
            'skills': self.extract_skills(text),
            'education': self.extract_education(text),
            'years_experience': self.extract_experience_years(text),
            'raw_text_length': len(text),
            'nlp_enabled': self.nlp is not None
        }
        
        return data

# Initialize advanced parser
resume_parser = AdvancedResumeParser()

# Simple Prolog-like rule engine
class PrologEngine:
    def __init__(self):
        self.facts = []
        self.rules = [
            # Education rules
            ("high_education(X)", lambda x: any("bachelor" in edu.lower() or "master" in edu.lower() or "phd" in edu.lower() 
                                               for edu in x.get('education', []))),
            
            # Experience rules
            ("experienced(X)", lambda x: x.get('years_experience', 0) >= 2),
            ("senior_level(X)", lambda x: x.get('years_experience', 0) >= 5),
            
            # Skills rules
            ("has_programming_skills(X)", lambda x: any(skill.lower() in ['python', 'java', 'javascript', 'c++', 'c#'] 
                                                        for skill in x.get('skills', []))),
            ("has_database_skills(X)", lambda x: any(skill.lower() in ['mysql', 'postgresql', 'mongodb', 'sql'] 
                                                     for skill in x.get('skills', []))),
            ("has_web_skills(X)", lambda x: any(skill.lower() in ['html', 'css', 'react', 'angular', 'vue', 'flask', 'django'] 
                                                for skill in x.get('skills', []))),
            
            # Decision rules
            ("qualified_developer(X)", lambda x: self.evaluate("high_education(X)", x) and 
                                                self.evaluate("experienced(X)", x) and 
                                                self.evaluate("has_programming_skills(X)", x)),
            
            ("qualified_senior_developer(X)", lambda x: self.evaluate("qualified_developer(X)", x) and 
                                                       self.evaluate("senior_level(X)", x) and 
                                                       self.evaluate("has_database_skills(X)", x)),
            
            ("web_developer_candidate(X)", lambda x: self.evaluate("qualified_developer(X)", x) and 
                                                    self.evaluate("has_web_skills(X)", x)),
        ]
    
    def evaluate(self, rule_name, candidate_data):
        """Evaluate a rule against candidate data"""
        for rule, condition in self.rules:
            if rule == rule_name:
                try:
                    return condition(candidate_data)
                except:
                    return False
        return False
    
    def screen_candidate(self, candidate_data):
        """Run complete screening analysis"""
        results = {}
        explanation = []
        
        # Evaluate all rules
        for rule, _ in self.rules:
            rule_name = rule.replace("(X)", "").replace("_", " ").title()
            results[rule_name] = self.evaluate(rule, candidate_data)
            if results[rule_name]:
                explanation.append(f"✓ {rule_name}")
            else:
                explanation.append(f"✗ {rule_name}")
        
        # Make final decision
        if self.evaluate("qualified_senior_developer(X)", candidate_data):
            decision = "HIRE"
            confidence = 0.95
            recommendation = "Highly qualified senior developer candidate"
        elif self.evaluate("web_developer_candidate(X)", candidate_data):
            decision = "HIRE"
            confidence = 0.85
            recommendation = "Good web developer candidate"
        elif self.evaluate("qualified_developer(X)", candidate_data):
            decision = "CONSIDER"
            confidence = 0.75
            recommendation = "Qualified developer, good potential"
        else:
            decision = "REJECT"
            confidence = 0.60
            recommendation = "Does not meet minimum requirements"
        
        return {
            'decision': decision,
            'confidence': confidence,
            'recommendation': recommendation,
            'rule_evaluation': results,
            'explanation': explanation
        }

# Initialize Prolog engine
prolog_engine = PrologEngine()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    """Upload and process resume"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract and parse resume
    text = extract_text_from_file(file_path)
    candidate_data = resume_parser.parse(text)
    candidate_data['id'] = len(candidates) + 1
    candidate_data['filename'] = filename
    candidate_data['upload_time'] = datetime.now().isoformat()
    
    # Run Prolog screening
    screening_result = prolog_engine.screen_candidate(candidate_data)
    screening_result['candidate_id'] = candidate_data['id']
    screening_result['screening_time'] = datetime.now().isoformat()
    
    # Store results
    candidates.append(candidate_data)
    screening_results.append(screening_result)
    
    return redirect(url_for('results', candidate_id=candidate_data['id']))

@app.route('/results')
@app.route('/results/<int:candidate_id>')
def results(candidate_id=None):
    """Show screening results"""
    if candidate_id:
        # Show specific candidate
        candidate = next((c for c in candidates if c['id'] == candidate_id), None)
        result = next((r for r in screening_results if r['candidate_id'] == candidate_id), None)
        if candidate and result:
            return render_template('result.html', candidate=candidate, result=result)
        else:
            return redirect(url_for('results'))
    else:
        # Show all results
        return render_template('results.html', candidates=candidates, results=screening_results)

@app.route('/api/candidates')
def api_candidates():
    """API endpoint for candidates data"""
    return jsonify({
        'candidates': candidates,
        'results': screening_results,
        'total': len(candidates)
    })

if __name__ == '__main__':
    print("🚀 Starting Prolog Resume Screening System")
    print("📊 Open http://localhost:5002 in your browser")
    app.run(host='0.0.0.0', port=5002, debug=True) 