#!/usr/bin/env python3
"""
Rule-Based Knowledge Base for Resume Screening System
Contains all the business logic and rules for resume evaluation
Separated from the main application for better maintainability
"""

from datetime import datetime
from typing import Dict, List, Any, Tuple


class SkillsKnowledgeBase:
    """
    Knowledge base for skills categorization and recognition
    """
    
    @staticmethod
    def get_skills_database() -> Dict[str, List[str]]:
        """
        Comprehensive skills database organized by categories
        Used by both rule-based and ML-enhanced skill extraction
        """
        return {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'scala', 'kotlin', 'swift', 'php', 'ruby', 'perl', 'r', 'matlab', 'julia',
                'dart', 'elixir', 'haskell', 'f#', 'clojure', 'erlang', 'lua', 'groovy'
            ],
            'web_frontend': [
                'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt.js', 'gatsby',
                'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'material-ui',
                'chakra-ui', 'styled-components', 'webpack', 'vite', 'parcel', 'jquery'
            ],
            'web_backend': [
                'django', 'flask', 'express', 'fastapi', 'spring', 'laravel', 'rails', 
                'node.js', 'asp.net', 'gin', 'echo', 'fiber', 'actix', 'rocket', 'phoenix',
                'nest.js', 'spring boot', 'symfony', 'sinatra', 'asp.net core'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sql server', 'sqlite', 'dynamodb', 'firebase', 'supabase',
                'neo4j', 'couchbase', 'influxdb', 'clickhouse', 'snowflake'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 
                'cloudflare', 'linode', 'vultr', 'oracle cloud', 'vercel', 'netlify'
            ],
            'devops': [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
                'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'helm',
                'istio', 'prometheus', 'grafana', 'elk stack', 'datadog'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'slack', 'trello', 'figma', 'photoshop', 
                'vs code', 'intellij', 'eclipse', 'vim', 'emacs', 'postman', 'insomnia'
            ],
            'ml_ai': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
                'scikit-learn', 'pandas', 'numpy', 'opencv', 'keras', 'hugging face',
                'transformers', 'spacy', 'nltk', 'gensim', 'xgboost', 'lightgbm'
            ],
            'data_tools': [
                'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'tableau', 'power bi',
                'looker', 'metabase', 'superset', 'jupyter', 'anaconda'
            ],
            'mobile': [
                'react native', 'flutter', 'ionic', 'xamarin', 'swift ui', 
                'jetpack compose', 'android studio', 'xcode', 'cordova'
            ]
        }
    
    @staticmethod
    def get_skill_formatting_rules() -> Dict[str, str]:
        """
        Rules for properly formatting skill names
        """
        return {
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
            'google cloud': 'Google Cloud Platform',
            'machine learning': 'Machine Learning',
            'deep learning': 'Deep Learning'
        }


class EducationKnowledgeBase:
    """
    Knowledge base for education level recognition and evaluation
    """
    
    @staticmethod
    def get_education_levels() -> Dict[str, List[str]]:
        """
        Education level keywords for recognition
        """
        return {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral', 'doctor of philosophy'],
            'master': ['master', 'msc', 'm.sc', 'ma', 'm.a', 'mba', 'm.b.a', 'ms', 'm.s', 'masters'],
            'bachelor': ['bachelor', 'bsc', 'b.sc', 'ba', 'b.a', 'bs', 'b.s', 'be', 'b.e', 'btech', 'b.tech', 'bachelors']
        }
    
    @staticmethod
    def get_education_score(education_level: str) -> int:
        """
        Get numerical score for education level
        """
        scores = {
            'phd': 100,
            'master': 80,
            'bachelor': 60,
            'none': 0
        }
        return scores.get(education_level.lower(), 0)


class ExperienceKnowledgeBase:
    """
    Knowledge base for experience evaluation and patterns
    """
    
    @staticmethod
    def get_experience_patterns() -> List[str]:
        """
        Regex patterns for extracting years of experience
        """
        return [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?\s+(?:in|with|of)',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr[s]?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?\s+(?:working|developing|programming)',
            r'over\s+(\d+)\s+years?\s+(?:of\s+)?(?:experience|in)',
            r'more\s+than\s+(\d+)\s+years?\s+(?:of\s+)?(?:experience|in)'
        ]
    
    @staticmethod
    def get_date_range_pattern() -> str:
        """
        Pattern for extracting date ranges from work history
        """
        return r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)'
    
    @staticmethod
    def calculate_experience_score(years: int, min_required: int, senior_threshold: int) -> int:
        """
        Calculate experience score based on years and thresholds
        """
        if years >= senior_threshold:
            return 100
        elif years >= min_required:
            return 70
        else:
            return max(0, (years / min_required) * 50)


class ScreeningRulesKnowledgeBase:
    """
    Knowledge base containing all screening rules and decision logic
    """
    
    @staticmethod
    def get_default_requirements() -> Dict[str, Any]:
        """
        Default requirements for screening
        """
        return {
            'min_education': 'bachelor',
            'min_experience': 2,
            'senior_experience': 5,
            'required_skills': ['Python', 'JavaScript', 'SQL'],
            'preferred_skills': ['Docker', 'AWS', 'React'],
            'job_title': 'Software Developer',
            'job_category': 'software_development',
            'education_weight': 20,
            'experience_weight': 30,
            'skills_weight': 40,
            'preferred_weight': 10
        }
    
    @staticmethod
    def get_decision_thresholds() -> Dict[str, float]:
        """
        Score thresholds for making decisions
        """
        return {
            'hire_threshold': 85.0,
            'consider_threshold': 70.0,
            'maybe_threshold': 50.0,
            'reject_threshold': 0.0
        }
    
    @staticmethod
    def get_confidence_mapping() -> Dict[str, Tuple[float, float]]:
        """
        Confidence score ranges for each decision type
        (min_confidence, max_confidence)
        """
        return {
            'HIRE': (0.85, 0.95),
            'CONSIDER': (0.70, 0.85),
            'MAYBE': (0.50, 0.65),
            'REJECT': (0.30, 0.50)
        }
    
    @staticmethod
    def evaluate_education_requirement(candidate_education: List[str], min_education: str) -> bool:
        """
        Rule: Check if candidate meets minimum education requirement
        """
        if min_education == 'none':
            return True
        
        education_text = ' '.join(candidate_education).lower()
        education_levels = EducationKnowledgeBase.get_education_levels()
        
        if min_education == 'bachelor':
            return any(level in education_text for level in 
                      education_levels['bachelor'] + education_levels['master'] + education_levels['phd'])
        elif min_education == 'master':
            return any(level in education_text for level in 
                      education_levels['master'] + education_levels['phd'])
        elif min_education == 'phd':
            return any(level in education_text for level in education_levels['phd'])
        
        return False
    
    @staticmethod
    def evaluate_experience_requirement(years_experience: int, min_experience: int) -> bool:
        """
        Rule: Check if candidate meets minimum experience requirement
        """
        return years_experience >= min_experience
    
    @staticmethod
    def evaluate_senior_level(years_experience: int, senior_threshold: int) -> bool:
        """
        Rule: Check if candidate qualifies as senior level
        """
        return years_experience >= senior_threshold
    
    @staticmethod
    def evaluate_required_skills(candidate_skills: List[str], required_skills: List[str], threshold: float = 0.7) -> bool:
        """
        Rule: Check if candidate has required skills (70% match threshold)
        """
        if not required_skills:
            return True
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        required_skills_lower = [skill.lower().strip() for skill in required_skills]
        
        matched_skills = sum(1 for req_skill in required_skills_lower 
                           if any(req_skill in cand_skill or cand_skill in req_skill 
                                 for cand_skill in candidate_skills_lower))
        
        return matched_skills >= (len(required_skills_lower) * threshold)
    
    @staticmethod
    def evaluate_preferred_skills(candidate_skills: List[str], preferred_skills: List[str]) -> bool:
        """
        Rule: Check if candidate has any preferred skills
        """
        if not preferred_skills:
            return True
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        preferred_skills_lower = [skill.lower().strip() for skill in preferred_skills]
        
        return any(pref_skill in cand_skill or cand_skill in pref_skill 
                  for pref_skill in preferred_skills_lower 
                  for cand_skill in candidate_skills_lower)
    
    @staticmethod
    def evaluate_qualified_candidate(candidate_data: Dict, requirements: Dict) -> bool:
        """
        Composite Rule: Check if candidate is qualified (meets basic requirements)
        """
        return (
            ScreeningRulesKnowledgeBase.evaluate_education_requirement(
                candidate_data.get('education', []), 
                requirements['min_education']
            ) and
            ScreeningRulesKnowledgeBase.evaluate_experience_requirement(
                candidate_data.get('years_experience', 0), 
                requirements['min_experience']
            ) and
            ScreeningRulesKnowledgeBase.evaluate_required_skills(
                candidate_data.get('skills', []), 
                requirements['required_skills']
            )
        )
    
    @staticmethod
    def evaluate_recommended_candidate(candidate_data: Dict, requirements: Dict) -> bool:
        """
        Composite Rule: Check if candidate is highly recommended
        """
        return (
            ScreeningRulesKnowledgeBase.evaluate_qualified_candidate(candidate_data, requirements) and
            ScreeningRulesKnowledgeBase.evaluate_senior_level(
                candidate_data.get('years_experience', 0), 
                requirements['senior_experience']
            ) and
            ScreeningRulesKnowledgeBase.evaluate_preferred_skills(
                candidate_data.get('skills', []), 
                requirements['preferred_skills']
            )
        )
    
    @staticmethod
    def calculate_detailed_scores(candidate_data: Dict, requirements: Dict) -> Dict[str, float]:
        """
        Calculate detailed scores for each evaluation category
        """
        scores = {}
        
        # Education score
        education_list = candidate_data.get('education', [])
        education_text = ' '.join(education_list).lower()
        education_levels = EducationKnowledgeBase.get_education_levels()
        
        if any(level in education_text for level in education_levels['phd']):
            scores['education'] = 100
        elif any(level in education_text for level in education_levels['master']):
            scores['education'] = 80
        elif any(level in education_text for level in education_levels['bachelor']):
            scores['education'] = 60
        else:
            scores['education'] = 0
        
        # Experience score
        years_exp = candidate_data.get('years_experience', 0)
        scores['experience'] = ExperienceKnowledgeBase.calculate_experience_score(
            years_exp, requirements['min_experience'], requirements['senior_experience']
        )
        
        # Required skills score
        candidate_skills = [skill.lower() for skill in candidate_data.get('skills', [])]
        required_skills = [skill.lower().strip() for skill in requirements['required_skills']]
        
        if required_skills:
            matched_skills = sum(1 for req_skill in required_skills 
                               if any(req_skill in cand_skill or cand_skill in req_skill 
                                     for cand_skill in candidate_skills))
            scores['skills'] = (matched_skills / len(required_skills)) * 100
        else:
            scores['skills'] = 100
        
        # Preferred skills score
        preferred_skills = [skill.lower().strip() for skill in requirements['preferred_skills']]
        if preferred_skills:
            matched_preferred = sum(1 for pref_skill in preferred_skills 
                                  if any(pref_skill in cand_skill or cand_skill in pref_skill 
                                        for cand_skill in candidate_skills))
            scores['preferred'] = (matched_preferred / len(preferred_skills)) * 100
        else:
            scores['preferred'] = 100
        
        return scores
    
    @staticmethod
    def calculate_weighted_score(detailed_scores: Dict[str, float], requirements: Dict) -> float:
        """
        Calculate final weighted score based on requirements
        """
        return (
            detailed_scores['education'] * (requirements['education_weight'] / 100) +
            detailed_scores['experience'] * (requirements['experience_weight'] / 100) +
            detailed_scores['skills'] * (requirements['skills_weight'] / 100) +
            detailed_scores['preferred'] * (requirements['preferred_weight'] / 100)
        )
    
    @staticmethod
    def make_decision(total_score: float, candidate_data: Dict, requirements: Dict) -> Tuple[str, float, str]:
        """
        Make final hiring decision based on score and rules
        Returns: (decision, confidence, recommendation)
        """
        thresholds = ScreeningRulesKnowledgeBase.get_decision_thresholds()
        confidence_mapping = ScreeningRulesKnowledgeBase.get_confidence_mapping()
        
        # Decision logic
        if (total_score >= thresholds['hire_threshold'] and 
            ScreeningRulesKnowledgeBase.evaluate_recommended_candidate(candidate_data, requirements)):
            decision = "HIRE"
        elif (total_score >= thresholds['consider_threshold'] and 
              ScreeningRulesKnowledgeBase.evaluate_qualified_candidate(candidate_data, requirements)):
            decision = "CONSIDER"
        elif total_score >= thresholds['maybe_threshold']:
            decision = "MAYBE"
        else:
            decision = "REJECT"
        
        # Calculate confidence
        min_conf, max_conf = confidence_mapping[decision]
        confidence = min(max_conf, total_score / 100)
        
        # Generate recommendation
        job_title = requirements['job_title']
        recommendations = {
            "HIRE": f"Highly recommended candidate for {job_title}",
            "CONSIDER": f"Good candidate for {job_title}, meets most requirements",
            "MAYBE": f"Potential candidate, some requirements met",
            "REJECT": "Does not meet minimum requirements"
        }
        
        return decision, confidence, recommendations[decision]
    
    @staticmethod
    def get_rule_evaluation_list() -> List[Tuple[str, str]]:
        """
        Get list of rules to evaluate for transparency
        """
        return [
            ("evaluate_education_requirement", "Meets Education Requirement"),
            ("evaluate_experience_requirement", "Meets Experience Requirement"),
            ("evaluate_senior_level", "Is Senior Level"),
            ("evaluate_required_skills", "Has Required Skills"),
            ("evaluate_preferred_skills", "Has Preferred Skills"),
            ("evaluate_qualified_candidate", "Qualified Candidate"),
            ("evaluate_recommended_candidate", "Recommended Candidate"),
        ]


class ContactExtractionKnowledgeBase:
    """
    Knowledge base for contact information extraction patterns
    """
    
    @staticmethod
    def get_email_patterns() -> List[str]:
        """
        Patterns for email extraction
        """
        return [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
    
    @staticmethod
    def get_phone_patterns() -> List[str]:
        """
        Patterns for phone number extraction
        """
        return [
            r'\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}'
        ]
    
    @staticmethod
    def get_name_exclusion_keywords() -> List[str]:
        """
        Keywords that indicate a name candidate should be excluded
        """
        return [
            'university', 'college', 'company', 'inc', 'corp', 'ltd', 
            'llc', 'organization', 'institute', 'school', 'academy'
        ]


# Export all knowledge bases for easy import
__all__ = [
    'SkillsKnowledgeBase',
    'EducationKnowledgeBase', 
    'ExperienceKnowledgeBase',
    'ScreeningRulesKnowledgeBase',
    'ContactExtractionKnowledgeBase'
] 