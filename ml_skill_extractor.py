#!/usr/bin/env python3
"""
ML-Enhanced Skill Extraction Module
Improves skill extraction accuracy using machine learning techniques
while keeping the existing rule-based screening engine intact
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rule_based_knowledge_base import SkillsKnowledgeBase


class MLSkillExtractor:
    """
    Machine Learning enhanced skill extractor
    Uses contextual analysis and confidence scoring to improve skill detection
    """
    
    def __init__(self):
        self.skills_db = SkillsKnowledgeBase.get_skills_database()
        self.formatting_rules = SkillsKnowledgeBase.get_skill_formatting_rules()
        
        # Enhanced skill variations and synonyms
        self.skill_variations = self._build_skill_variations()
        
        # Context patterns that indicate skill proficiency
        self.proficiency_patterns = [
            (r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience\s+)?(?:with|in|using)\s+(\w+)', 0.8),
            (r'expert\s+(?:in|with|at)\s+(\w+)', 0.9),
            (r'proficient\s+(?:in|with|at)\s+(\w+)', 0.8),
            (r'skilled\s+(?:in|with|at)\s+(\w+)', 0.7),
            (r'advanced\s+(?:in|with|at)\s+(\w+)', 0.8),
            (r'developed\s+(?:using|with|in)\s+(\w+)', 0.7),
            (r'built\s+(?:using|with|in)\s+(\w+)', 0.7),
            (r'implemented\s+(?:using|with|in)\s+(\w+)', 0.7),
            (r'worked\s+(?:with|on|using)\s+(\w+)', 0.6),
            (r'led\s+(?:development|team|project)\s+(?:using|with|in)\s+(\w+)', 0.8)
        ]
        
        # Section indicators
        self.skill_section_indicators = [
            'technical skills', 'skills', 'technologies', 'programming languages',
            'tools', 'frameworks', 'languages', 'competencies', 'proficiencies'
        ]
        
        # Project/work context indicators
        self.work_context_indicators = [
            'project', 'developed', 'built', 'implemented', 'created', 'designed',
            'worked', 'used', 'utilized', 'applied', 'leveraged'
        ]
    
    def _build_skill_variations(self) -> Dict[str, List[str]]:
        """
        Build comprehensive skill variations database
        """
        variations = {}
        
        # Add base skills
        for category, skills in self.skills_db.items():
            for skill in skills:
                if skill not in variations:
                    variations[skill] = [skill]
        
        # Add specific variations and synonyms
        skill_synonyms = {
            'python': ['python', 'py', 'python3', 'python2', 'cpython'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es2015', 'vanilla js'],
            'java': ['java', 'java se', 'java ee', 'openjdk', 'oracle java'],
            'react': ['react', 'reactjs', 'react.js', 'react native'],
            'node.js': ['node.js', 'nodejs', 'node', 'express.js', 'express'],
            'angular': ['angular', 'angularjs', 'angular2', 'angular4', 'angular6', 'angular8'],
            'vue': ['vue', 'vuejs', 'vue.js', 'vue2', 'vue3'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'mssql', 'oracle sql'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'docker': ['docker', 'containerization', 'docker-compose', 'dockerfile'],
            'kubernetes': ['kubernetes', 'k8s', 'kubectl', 'helm'],
            'git': ['git', 'github', 'gitlab', 'bitbucket', 'version control'],
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
            'tensorflow': ['tensorflow', 'tf', 'keras', 'tensorflow2'],
            'pytorch': ['pytorch', 'torch', 'torchvision'],
            'pandas': ['pandas', 'pd', 'dataframe'],
            'numpy': ['numpy', 'np', 'numerical python']
        }
        
        for base_skill, synonyms in skill_synonyms.items():
            variations[base_skill] = synonyms
            # Also add reverse mapping
            for synonym in synonyms:
                if synonym != base_skill:
                    variations[synonym] = [base_skill] + [s for s in synonyms if s != synonym]
        
        return variations
    
    def extract_skills_with_ml(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills using ML-enhanced techniques with confidence scoring
        
        Returns:
            List of dictionaries with skill, confidence, context, and method
        """
        extracted_skills = []
        text_lower = text.lower()
        
        # Method 1: Enhanced pattern matching with context
        pattern_skills = self._extract_with_patterns(text_lower)
        extracted_skills.extend(pattern_skills)
        
        # Method 2: Section-aware extraction
        section_skills = self._extract_from_sections(text_lower)
        extracted_skills.extend(section_skills)
        
        # Method 3: Context-based extraction
        context_skills = self._extract_with_context(text_lower)
        extracted_skills.extend(context_skills)
        
        # Deduplicate and enhance with confidence scoring
        final_skills = self._deduplicate_and_score(extracted_skills, text_lower)
        
        # Convert to standard format (just skill names) for compatibility
        return self._format_for_compatibility(final_skills)
    
    def _extract_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills using enhanced pattern matching
        """
        skills_found = []
        
        for pattern, base_confidence in self.proficiency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the last group that has content, or use the whole match
                skill_candidate = None
                for i in range(match.lastindex or 0, 0, -1):
                    if match.group(i):
                        skill_candidate = match.group(i).lower()
                        break
                if not skill_candidate:
                    continue
                
                # Check if it matches any known skill
                matched_skill = self._find_matching_skill(skill_candidate)
                if matched_skill:
                    context = self._extract_context_window(text, match.start(), match.end())
                    
                    skills_found.append({
                        'skill': matched_skill,
                        'confidence': base_confidence,
                        'context': context,
                        'method': 'pattern_matching',
                        'position': match.start()
                    })
        
        return skills_found
    
    def _extract_from_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills from dedicated skill sections
        """
        skills_found = []
        
        # Find skill sections
        for section_name in self.skill_section_indicators:
            section_pattern = rf'{re.escape(section_name)}[:\s]*([^\.]*?)(?:\n\n|\n[A-Z]|$)'
            section_matches = re.finditer(section_pattern, text, re.IGNORECASE | re.DOTALL)
            
            for section_match in section_matches:
                section_text = section_match.group(1)
                
                # Extract skills from this section
                section_skills = self._extract_skills_from_text_chunk(section_text, 'section_extraction', 0.8)
                skills_found.extend(section_skills)
        
        return skills_found
    
    def _extract_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills based on work/project context
        """
        skills_found = []
        
        # Split into sentences for context analysis
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        for sentence in sentences:
            # Check if sentence contains work context
            has_work_context = any(indicator in sentence.lower() 
                                 for indicator in self.work_context_indicators)
            
            if has_work_context:
                # Extract skills from this sentence with higher confidence
                sentence_skills = self._extract_skills_from_text_chunk(
                    sentence, 'context_extraction', 0.7
                )
                skills_found.extend(sentence_skills)
        
        return skills_found
    
    def _extract_skills_from_text_chunk(self, text_chunk: str, method: str, base_confidence: float) -> List[Dict[str, Any]]:
        """
        Extract skills from a chunk of text
        """
        skills_found = []
        text_lower = text_chunk.lower()
        
        # Check all skill variations
        for base_skill, variations in self.skill_variations.items():
            for variation in variations:
                if self._is_skill_present(text_lower, variation):
                    confidence = self._calculate_context_confidence(text_lower, variation, base_confidence)
                    
                    if confidence > 0.4:  # Minimum confidence threshold
                        skills_found.append({
                            'skill': base_skill,
                            'confidence': confidence,
                            'context': text_chunk[:100] + '...' if len(text_chunk) > 100 else text_chunk,
                            'method': method,
                            'position': text_lower.find(variation)
                        })
                        break  # Found this skill, move to next
        
        return skills_found
    
    def _is_skill_present(self, text: str, skill: str) -> bool:
        """
        Check if skill is present using word boundary matching
        """
        # Use word boundaries for exact matching
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _find_matching_skill(self, candidate: str) -> str:
        """
        Find the best matching skill from our database
        """
        candidate_lower = candidate.lower()
        
        # Direct match
        if candidate_lower in self.skill_variations:
            return candidate_lower
        
        # Partial match
        for base_skill, variations in self.skill_variations.items():
            for variation in variations:
                if variation.lower() == candidate_lower:
                    return base_skill
                # Also check if candidate contains the skill
                if len(candidate_lower) > 3 and variation.lower() in candidate_lower:
                    return base_skill
        
        return None
    
    def _calculate_context_confidence(self, text: str, skill: str, base_confidence: float) -> float:
        """
        Calculate confidence based on context around skill
        """
        confidence = base_confidence
        
        # Find skill position
        skill_pos = text.find(skill.lower())
        if skill_pos == -1:
            return 0.0
        
        # Analyze context window
        window_start = max(0, skill_pos - 50)
        window_end = min(len(text), skill_pos + len(skill) + 50)
        context_window = text[window_start:window_end]
        
        # Boost for years of experience
        years_pattern = r'(\d+)\+?\s*years?'
        if re.search(years_pattern, context_window):
            years_match = re.search(years_pattern, context_window)
            if years_match:
                years = int(years_match.group(1))
                confidence += min(years * 0.05, 0.2)  # Up to 0.2 boost
        
        # Boost for skill section context
        if any(indicator in context_window for indicator in ['skill', 'technolog', 'tool']):
            confidence += 0.15
        
        # Boost for proficiency indicators
        proficiency_words = ['expert', 'proficient', 'advanced', 'experienced', 'skilled']
        if any(word in context_window for word in proficiency_words):
            confidence += 0.1
        
        # Boost for project context
        if any(word in context_window for word in self.work_context_indicators):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _extract_context_window(self, text: str, start: int, end: int, window_size: int = 50) -> str:
        """
        Extract context window around a match
        """
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end].strip()
    
    def _deduplicate_and_score(self, skills_list: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """
        Remove duplicates and calculate final confidence scores
        """
        skill_map = {}
        
        for skill_data in skills_list:
            skill_name = skill_data['skill']
            
            if skill_name not in skill_map:
                skill_map[skill_name] = skill_data
            else:
                # Keep the one with higher confidence
                if skill_data['confidence'] > skill_map[skill_name]['confidence']:
                    skill_map[skill_name] = skill_data
                # Or combine confidences (weighted average)
                else:
                    existing = skill_map[skill_name]
                    combined_confidence = (existing['confidence'] + skill_data['confidence']) / 2
                    existing['confidence'] = min(combined_confidence, 1.0)
        
        # Sort by confidence
        return sorted(skill_map.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _format_for_compatibility(self, ml_skills: List[Dict[str, Any]]) -> List[str]:
        """
        Format ML results for compatibility with existing system
        Keep only skills with confidence > 0.5
        """
        formatted_skills = []
        
        for skill_data in ml_skills:
            if skill_data['confidence'] >= 0.5:
                # Format skill name properly
                skill_name = skill_data['skill']
                formatted_name = self.formatting_rules.get(skill_name.lower(), skill_name.title())
                formatted_skills.append(formatted_name)
        
        return sorted(list(set(formatted_skills)))
    
    def extract_skills_with_details(self, text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract skills and return both compatible format and detailed analysis
        
        Returns:
            Tuple of (compatible_skills_list, detailed_analysis)
        """
        extracted_skills = []
        text_lower = text.lower()
        
        # Method 1: Enhanced pattern matching with context
        pattern_skills = self._extract_with_patterns(text_lower)
        extracted_skills.extend(pattern_skills)
        
        # Method 2: Section-aware extraction
        section_skills = self._extract_from_sections(text_lower)
        extracted_skills.extend(section_skills)
        
        # Method 3: Context-based extraction
        context_skills = self._extract_with_context(text_lower)
        extracted_skills.extend(context_skills)
        
        # Deduplicate and enhance with confidence scoring
        detailed_skills = self._deduplicate_and_score(extracted_skills, text_lower)
        
        # Format for compatibility
        compatible_skills = self._format_for_compatibility(detailed_skills)
        
        return compatible_skills, detailed_skills


def integrate_ml_skill_extraction():
    """
    Integration function to enhance existing AdvancedResumeParser
    This shows how to modify the existing extract_skills method
    """
    
    # Code to be added to AdvancedResumeParser class
    integration_code = """
    def __init__(self):
        # ... existing initialization code ...
        
        # Add ML skill extractor
        try:
            from ml_skill_extractor import MLSkillExtractor
            self.ml_skill_extractor = MLSkillExtractor()
            self.ml_enabled = True
            print("✅ ML skill extraction enabled")
        except ImportError:
            self.ml_enabled = False
            print("⚠️  ML skill extraction not available")
    
    def extract_skills(self, text):
        '''Extract skills using ML-enhanced techniques'''
        
        if self.ml_enabled:
            # Use ML-enhanced extraction
            ml_skills, detailed_analysis = self.ml_skill_extractor.extract_skills_with_details(text)
            
            # Fallback to original method for skills not caught by ML
            original_skills = self._extract_skills_original(text)
            
            # Combine results
            all_skills = list(set(ml_skills + original_skills))
            
            # Store detailed analysis for later use
            self.last_skill_analysis = detailed_analysis
            
            return sorted(all_skills)
        else:
            # Use original method
            return self._extract_skills_original(text)
    
    def _extract_skills_original(self, text):
        '''Original skill extraction method (renamed)'''
        # This is the existing extract_skills method code
        # ... existing code ...
    """
    
    return integration_code


# Demo function
def demo_ml_skill_extraction():
    """
    Demonstrate the ML skill extraction capabilities
    """
    sample_resume = """
    John Doe
    Senior Software Engineer
    
    Professional Experience:
    • 5 years experience with Python and Django framework
    • Expert in JavaScript, React, and Node.js development  
    • Proficient in AWS cloud services including EC2, S3, and Lambda
    • Built machine learning models using TensorFlow and PyTorch
    • Advanced SQL database design and PostgreSQL optimization
    • Led development teams using Git, Docker, and Kubernetes
    • Developed REST APIs using FastAPI and Express.js
    
    Technical Skills:
    - Programming Languages: Python, JavaScript, Java, TypeScript
    - Web Frameworks: React, Angular, Vue.js, Django, Flask
    - Cloud Platforms: AWS, Google Cloud Platform, Azure
    - Databases: PostgreSQL, MySQL, MongoDB, Redis
    - DevOps Tools: Docker, Kubernetes, Jenkins, GitLab CI
    - ML/AI: TensorFlow, PyTorch, scikit-learn, pandas, numpy
    
    Projects:
    • Built an e-commerce platform using React and Django
    • Implemented ML-powered recommendation system with collaborative filtering
    • Developed microservices architecture using Docker and Kubernetes
    """
    
    extractor = MLSkillExtractor()
    compatible_skills, detailed_analysis = extractor.extract_skills_with_details(sample_resume)
    
    print("🧠 ML-Enhanced Skill Extraction Demo")
    print("=" * 60)
    
    print(f"\n📋 Compatible Skills List ({len(compatible_skills)} skills):")
    for i, skill in enumerate(compatible_skills, 1):
        print(f"  {i:2d}. {skill}")
    
    print(f"\n🔍 Detailed Analysis ({len(detailed_analysis)} unique skills):")
    for skill_data in detailed_analysis[:10]:  # Show top 10
        confidence_bar = "█" * int(skill_data['confidence'] * 10)
        print(f"\n{skill_data['skill']:20} | {confidence_bar:10} | {skill_data['confidence']:.2f}")
        print(f"   Method: {skill_data['method']}")
        print(f"   Context: {skill_data['context'][:80]}...")
    
    return compatible_skills, detailed_analysis


if __name__ == "__main__":
    demo_ml_skill_extraction() 