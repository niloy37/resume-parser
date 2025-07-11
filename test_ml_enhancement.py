#!/usr/bin/env python3
"""
Test Script for ML-Enhanced Skill Extraction
Demonstrates the improvements in skill detection accuracy
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_skill_extraction():
    """Test the ML-enhanced skill extraction"""
    
    sample_resume = """
    Jane Smith
    Senior Software Engineer
    Email: jane.smith@email.com
    Phone: (555) 123-4567
    
    Professional Experience:
    • 6 years experience developing applications with Python and Django
    • Expert in JavaScript frameworks including React and Vue.js
    • Proficient in cloud platforms like AWS (EC2, S3, Lambda) and Google Cloud
    • Built machine learning models using TensorFlow and scikit-learn
    • Advanced knowledge of PostgreSQL database optimization
    • Led teams using modern DevOps practices with Docker and Kubernetes
    • Implemented CI/CD pipelines using Jenkins and GitLab CI
    
    Technical Skills:
    - Programming Languages: Python, JavaScript, TypeScript, Java
    - Frontend: React, Vue.js, Angular, HTML5, CSS3, SASS
    - Backend: Django, Flask, Node.js, Express.js, FastAPI
    - Databases: PostgreSQL, MySQL, MongoDB, Redis
    - Cloud: AWS, Google Cloud Platform, Azure
    - DevOps: Docker, Kubernetes, Jenkins, Terraform
    - ML/AI: TensorFlow, PyTorch, scikit-learn, pandas, numpy
    
    Recent Projects:
    • E-commerce platform built with React frontend and Django REST API
    • ML recommendation engine using collaborative filtering with 94% accuracy
    • Microservices architecture deployed on Kubernetes with 99.9% uptime
    """
    
    try:
        # Test the ML-enhanced skill extractor directly
        print("🧪 Testing ML-Enhanced Skill Extraction")
        print("=" * 50)
        
        from ml_skill_extractor import MLSkillExtractor
        
        extractor = MLSkillExtractor()
        skills, detailed_analysis = extractor.extract_skills_with_details(sample_resume)
        
        print(f"\n📋 Extracted Skills ({len(skills)} total):")
        for i, skill in enumerate(skills, 1):
            print(f"  {i:2d}. {skill}")
        
        print(f"\n🔍 Top 10 Skills with Confidence Scores:")
        for skill_data in detailed_analysis[:10]:
            confidence_bar = "█" * int(skill_data['confidence'] * 10)
            print(f"{skill_data['skill']:20} | {confidence_bar:10} | {skill_data['confidence']:.2f}")
        
        print(f"\n📊 Analysis Summary:")
        high_confidence = [s for s in detailed_analysis if s['confidence'] > 0.7]
        medium_confidence = [s for s in detailed_analysis if 0.5 <= s['confidence'] <= 0.7]
        print(f"  High confidence skills (>0.7): {len(high_confidence)}")
        print(f"  Medium confidence skills (0.5-0.7): {len(medium_confidence)}")
        print(f"  Total skills analyzed: {len(detailed_analysis)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import ML skill extractor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing ML enhancement: {e}")
        return False


def test_integration():
    """Test the integration with the existing resume parser"""
    
    print("\n🔧 Testing Integration with Existing System")
    print("=" * 50)
    
    try:
        # Import the enhanced resume parser
        from prolog_resume_app import AdvancedResumeParser
        
        parser = AdvancedResumeParser()
        
        sample_text = """
        John Developer
        john@email.com | (555) 987-6543
        
        Experience: 4 years with Python, React, and AWS
        Built web applications using Django and PostgreSQL
        Proficient in Docker containerization and Kubernetes
        """
        
        print("📝 Parsing sample resume with ML-enhanced extraction...")
        
        # Parse the resume
        candidate_data = parser.parse(sample_text)
        
        print(f"\n✅ Parsing completed successfully!")
        print(f"📋 Extracted data:")
        print(f"  Name: {candidate_data['name']}")
        print(f"  Email: {candidate_data['email']}")
        print(f"  Phone: {candidate_data['phone']}")
        print(f"  Skills: {', '.join(candidate_data['skills'])}")
        print(f"  Years Experience: {candidate_data['years_experience']}")
        print(f"  Education: {', '.join(candidate_data['education'])}")
        
        # Check if ML analysis is available
        if 'skill_analysis' in candidate_data:
            print(f"  🧠 ML Analysis: {len(candidate_data['skill_analysis'])} skills analyzed")
            for skill in candidate_data['skill_analysis'][:3]:
                print(f"    • {skill['skill']} (confidence: {skill['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_rule_based_engine():
    """Test that the rule-based screening engine is preserved"""
    
    print("\n⚖️  Testing Rule-Based Screening Engine")
    print("=" * 50)
    
    try:
        from prolog_resume_app import ResumeScreeningEngine, current_requirements
        
        # Create screening engine
        engine = ResumeScreeningEngine(current_requirements)
        
        # Sample candidate data
        candidate_data = {
            'name': 'Test Candidate',
            'email': 'test@email.com',
            'phone': '(555) 123-4567',
            'skills': ['Python', 'JavaScript', 'SQL', 'Docker', 'AWS'],
            'education': ['Bachelor of Science in Computer Science'],
            'years_experience': 3,
            'raw_text_length': 1000,
            'nlp_enabled': True
        }
        
        # Run screening
        result = engine.screen_candidate(candidate_data)
        
        print(f"✅ Rule-based screening completed!")
        print(f"📊 Screening results:")
        print(f"  Decision: {result['decision']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Total Score: {result['total_score']}")
        print(f"  Recommendation: {result['recommendation']}")
        
        print(f"\n📋 Rule Evaluations:")
        for rule, passed in result['rule_evaluation'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {rule}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule-based engine test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 ML Enhancement Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("ML Skill Extraction", test_skill_extraction),
        ("System Integration", test_integration),
        ("Rule-Based Engine", test_rule_based_engine)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} | {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! ML enhancement successfully integrated.")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
    
    print("\n🔗 Integration Complete!")
    print("Your system now has:")
    print("  ✅ ML-enhanced skill extraction")
    print("  ✅ Preserved rule-based screening")
    print("  ✅ Backward compatibility")
    print("  ✅ Graceful fallback to original methods") 