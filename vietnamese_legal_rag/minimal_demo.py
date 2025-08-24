
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
    
    print("\n" + "="*60)
    print("🎯 MINIMAL DEMO VIETNAMESE LEGAL RAG SYSTEM")
    print("="*60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Demo {i} ---")
        rag_system.query(question)
        print("-" * 40)
    
    print("\n👋 Demo hoàn tất!")

if __name__ == "__main__":
    run_minimal_demo()
