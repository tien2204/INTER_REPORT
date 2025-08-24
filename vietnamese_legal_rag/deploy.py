#!/usr/bin/env python3
"""
Vietnamese Legal RAG System Deployment Script
Helps deploy and run the system in different environments
"""

import os
import sys
import subprocess
import json

def check_environment():
    """Check if we're running in Kaggle, Colab, or local environment"""
    if 'KAGGLE_CONTAINER_NAME' in os.environ:
        return 'kaggle'
    elif 'COLAB_GPU' in os.environ:
        return 'colab'
    else:
        return 'local'

def install_requirements():
    """Install required packages"""
    env = check_environment()
    print(f"🔧 Detected environment: {env}")
    
    # Basic requirements for all environments
    basic_packages = [
        "transformers",
        "torch",
        "datasets",
        "langchain",
        "langchain-community", 
        "langchain-core",
        "weaviate-client",
        "sentence-transformers",
        "rank_bm25",
        "accelerate",
        "bitsandbytes",
        "soundfile",
        "numpy",
        "pandas"
    ]
    
    if env in ['kaggle', 'colab']:
        # Use pip install with quiet flag for notebooks
        for package in basic_packages:
            try:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], 
                             check=True, capture_output=True)
                print(f"✅ {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")
    else:
        # Local environment - install from requirements.txt if available
        if os.path.exists('requirements.txt'):
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                             check=True)
                print("✅ Installed from requirements.txt")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install requirements: {e}")
        else:
            print("❌ requirements.txt not found")

def create_minimal_demo():
    """Create a minimal demo that works without heavy models"""
    
    demo_code = """
# Minimal Demo for Vietnamese Legal RAG System
import sys
import warnings
warnings.filterwarnings('ignore')

class MinimalVietnameseLegalRAG:
    '''Lightweight demo version without heavy model dependencies'''
    
    def __init__(self):
        print("🚀 Khởi tạo Minimal Vietnamese Legal RAG System...")
        self.sample_data = [
            {
                "question": "Tù chung thân là gì?",
                "answer": "Tù chung thân là hình phạt tù không có thời hạn, được áp dụng cho những tội phạm đặc biệt nghiêm trọng. Người bị kết án tù chung thân có thể được xem xét giảm án sau khi chấp hành ít nhất 20 năm tù.",
                "source": "Bộ luật Hình sự Việt Nam"
            },
            {
                "question": "Luật dân sự quy định gì?",
                "answer": "Luật Dân sự quy định về quyền và nghĩa vụ của công dân trong các quan hệ dân sự như quyền sở hữu, quyền thừa kế, hợp đồng mua bán.",
                "source": "Bộ luật Dân sự Việt Nam"
            },
            {
                "question": "Quyền bào chữa là gì?",
                "answer": "Quyền bào chữa là quyền cơ bản của người bị buộc tội. Mọi người đều có quyền tự bào chữa hoặc nhờ luật sư bào chữa khi bị truy cứu trách nhiệm hình sự.",
                "source": "Hiến pháp Việt Nam"
            }
        ]
        print("✅ Hệ thống demo đã sẵn sàng!")
    
    def query(self, question):
        '''Simple query matching for demo purposes'''
        print(f"❓ Câu hỏi: {question}")
        
        # Simple keyword matching
        best_match = None
        best_score = 0
        
        for item in self.sample_data:
            # Simple scoring based on common words
            common_words = set(question.lower().split()) & set(item['question'].lower().split())
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_match:
            print(f"📝 Câu trả lời: {best_match['answer']}")
            print(f"📚 Nguồn: {best_match['source']}")
            return best_match
        else:
            answer = "Xin lỗi, tôi chưa có thông tin về câu hỏi này. Vui lòng thử với câu hỏi khác về pháp luật Việt Nam."
            print(f"📝 Câu trả lời: {answer}")
            return {"answer": answer, "source": "System"}

def run_minimal_demo():
    '''Run the minimal demo'''
    rag_system = MinimalVietnameseLegalRAG()
    
    demo_questions = [
        "Tù chung thân là gì?",
        "Luật dân sự quy định gì?", 
        "Quyền bào chữa là gì?",
        "Hình phạt tử hình có được áp dụng không?"
    ]
    
    print("\\n" + "="*60)
    print("🎯 MINIMAL DEMO VIETNAMESE LEGAL RAG SYSTEM")
    print("="*60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\\n--- Demo {i} ---")
        rag_system.query(question)
        print("-" * 40)
    
    print("\\n👋 Demo hoàn tất!")

if __name__ == "__main__":
    run_minimal_demo()
"""
    
    with open('minimal_demo.py', 'w', encoding='utf-8') as f:
        f.write(demo_code)
    
    print("✅ Created minimal_demo.py")

def deploy():
    """Main deployment function"""
    print("🚀 VIETNAMESE LEGAL RAG SYSTEM DEPLOYMENT")
    print("="*60)
    
    env = check_environment()
    print(f"Environment: {env}")
    
    # Step 1: Check files
    required_files = [
        'vietnamese_legal_rag_system.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        print("Please ensure all required files are present.")
        return False
    
    # Step 2: Create minimal demo
    create_minimal_demo()
    
    # Step 3: Installation instructions
    print("\n📋 INSTALLATION INSTRUCTIONS")
    print("-" * 40)
    
    if env == 'kaggle':
        print("For Kaggle:")
        print("1. Run this in a code cell:")
        print("   !pip install -q transformers torch datasets langchain sentence-transformers weaviate-client")
        print("2. Copy vietnamese_legal_rag_system.py content to a cell")
        print("3. Run: exec(open('vietnamese_legal_rag_system.py').read())")
        
    elif env == 'colab':
        print("For Google Colab:")
        print("1. Run this in a cell:")
        print("   !pip install transformers torch datasets langchain sentence-transformers weaviate-client")
        print("2. Upload vietnamese_legal_rag_system.py")
        print("3. Run: exec(open('vietnamese_legal_rag_system.py').read())")
        
    else:
        print("For Local Environment:")
        print("1. Install requirements:")
        print("   pip install -r requirements.txt")
        print("2. Run the system:")
        print("   python vietnamese_legal_rag_system.py")
    
    # Step 4: Quick test option
    print(f"\n🧪 QUICK TEST")
    print("-" * 40)
    print("To test basic functionality without heavy models:")
    print("  python minimal_demo.py")
    
    print("\n✅ Deployment preparation complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'install':
            install_requirements()
        elif command == 'demo':
            create_minimal_demo()
            print("Run: python minimal_demo.py")
        elif command == 'test':
            exec(open('test_structure.py').read())
        else:
            print("Usage: python deploy.py [install|demo|test]")
    else:
        deploy()