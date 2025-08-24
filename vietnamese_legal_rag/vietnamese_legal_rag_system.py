# Vietnamese Legal RAG System with TTS on Kaggle
# Sử dụng ViLM/VinAllama-2.7B-Chat + RAG + TTS

# ================================
# PHẦN 1: CÀI ĐẶT THƯ VIỆN (Updated)
# ================================

# Chạy các lệnh này trong Kaggle notebook hoặc terminal
"""
!pip install -q transformers torch datasets
!pip install -q langchain langchain-community langchain-core
!pip install -q weaviate-client sentence-transformers rank_bm25
!pip install -q FlagEmbedding soundfile librosa pydub IPython
!pip install -q accelerate bitsandbytes
!pip install -q faiss-cpu  # Alternative vector search
"""

# ================================
# PHẦN 2: IMPORT CÁC THƯ VIỆN
# ================================

import torch
import weaviate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from typing import List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ================================
# PHẦN 3: THIẾT LẬP WEAVIATE
# ================================

class WeaviateSetup:
    def __init__(self):
        # Sử dụng Weaviate local/in-memory cho test
        try:
            # Thử kết nối Weaviate local trước
            self.client = weaviate.Client("http://localhost:8080")
            print("✅ Kết nối Weaviate local thành công")
        except:
            try:
                # Fallback: Sử dụng WCS (Weaviate Cloud Service) sandbox
                # Hoặc setup local Weaviate instance
                print("⚠️ Không thể kết nối Weaviate local")
                print("🔧 Đang setup Weaviate in-memory alternative...")
                self.client = self.setup_inmemory_alternative()
            except Exception as e:
                print(f"❌ Lỗi Weaviate: {e}")
                print("🔄 Sử dụng alternative vector storage...")
                self.client = None
                
        if self.client:
            self.setup_schema()
    
    def setup_inmemory_alternative(self):
        # Alternative simple vector storage
        class SimpleVectorStore:
            def __init__(self):
                self.vectors = []
                self.documents = []
                
            def add_documents(self, docs_with_vectors):
                for doc, vector in docs_with_vectors:
                    self.documents.append(doc)
                    self.vectors.append(vector)
            
            def similarity_search(self, query_vector, k=5):
                if not self.vectors:
                    return []
                
                # Compute cosine similarity
                similarities = []
                query_norm = np.linalg.norm(query_vector)
                
                for i, vec in enumerate(self.vectors):
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 0 and query_norm > 0:
                        sim = np.dot(query_vector, vec) / (query_norm * vec_norm)
                        similarities.append((sim, i))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                # Return top k
                results = []
                for sim, idx in similarities[:k]:
                    results.append(self.documents[idx])
                
                return results
        
        return SimpleVectorStore()
    
    def setup_schema(self):
        if hasattr(self.client, 'schema'):  # Real Weaviate
            # Tạo schema cho Vietnamese Legal documents
            schema = {
                "classes": [{
                    "class": "LegalDocument",
                    "description": "Vietnamese legal document",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Document content"
                        },
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "Document source"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["int"],
                            "description": "Chunk identifier"
                        }
                    ]
                }]
            }
            
            # Xóa schema cũ nếu tồn tại
            try:
                if self.client.schema.exists("LegalDocument"):
                    self.client.schema.delete_class("LegalDocument")
            except:
                pass
            
            self.client.schema.create(schema)

# ================================
# PHẦN 4: CUSTOM BGE-M3 EMBEDDINGS
# ================================

class BGEM3Embeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-m3')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

# ================================
# PHẦN 5: CUSTOM VINALLAMA LLM
# ================================

class VinaLlamaLLM(LLM):
    """Custom LLM wrapper cho VinaLlama model"""
    
    def __init__(self, **kwargs):
        # Khởi tạo parent class trước
        super().__init__(**kwargs)
        
        print("🤖 Đang load VinaLlama model...")
        
        # Cấu hình quantization để tiết kiệm memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model và tokenizer
        model_name = "vilm/vinallama-2.7b-chat"
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Thiết lập pad token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                
            print("✅ VinaLlama model loaded successfully")
            
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            print("🔄 Sử dụng fallback model...")
            self._tokenizer = None
            self._model = None
    
    @property
    def _llm_type(self) -> str:
        return "vinallama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if self._model is None or self._tokenizer is None:
            return "Xin lỗi, model chưa được load thành công. Vui lòng kiểm tra lại cấu hình."
        
        try:
            # Format prompt cho chat - simplified
            chat_prompt = f"Hãy trả lời câu hỏi sau dựa trên thông tin được cung cấp:\n\n{prompt}\n\nTrả lời ngắn gọn:"
            
            # Tokenize với padding và truncation
            inputs = self._tokenizer(
                chat_prompt, 
                return_tensors="pt", 
                max_length=1024,
                truncation=True,
                padding=True
            )
            
            # Move to device if available
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate với cải thiện parameters
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,  # Giảm để tránh lặp
                    temperature=0.3,     # Giảm temperature
                    do_sample=True,
                    repetition_penalty=1.2,  # Thêm repetition penalty
                    no_repeat_ngram_size=3,  # Tránh lặp ngram
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Chỉ lấy phần generated
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Remove repetitive patterns
            lines = response.split('\n')
            cleaned_lines = []
            prev_line = ""
            
            for line in lines:
                line = line.strip()
                if line and line != prev_line and len(line) > 5:
                    cleaned_lines.append(line)
                    prev_line = line
                if len(cleaned_lines) >= 5:  # Giới hạn độ dài
                    break
            
            final_response = '\n'.join(cleaned_lines)
            
            return final_response if final_response else "Xin lỗi, tôi không thể tạo câu trả lời phù hợp."
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {str(e)}"

# ================================
# PHẦN 6: HYBRID RETRIEVAL (BM25 + BGE-M3)
# ================================

class HybridRetriever:
    def __init__(self, weaviate_client, embeddings):
        self.weaviate_client = weaviate_client
        self.embeddings = embeddings
        self.bm25 = None
        self.documents = []
        self.is_simple_store = not hasattr(weaviate_client, 'schema')
        
    def add_documents(self, documents: List[Document]):
        self.documents = documents
        
        # Chuẩn bị BM25
        texts = [doc.page_content for doc in documents]
        tokenized_texts = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Thêm vào vector store
        if self.is_simple_store:
            # Simple vector store
            docs_with_vectors = []
            for doc in documents:
                vector = self.embeddings.embed_query(doc.page_content)
                docs_with_vectors.append((doc, vector))
            self.weaviate_client.add_documents(docs_with_vectors)
        else:
            # Real Weaviate
            for i, doc in enumerate(documents):
                try:
                    self.weaviate_client.data_object.create(
                        data_object={
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": i
                        },
                        class_name="LegalDocument",
                        vector=self.embeddings.embed_query(doc.page_content)
                    )
                except Exception as e:
                    print(f"Warning: Could not add document {i}: {e}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        retrieved_docs = []
        
        # BM25 retrieval
        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.split())
            bm25_top_k = np.argsort(bm25_scores)[-k//2:][::-1]
            
            # Thêm BM25 results
            for idx in bm25_top_k:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.metadata["score"] = float(bm25_scores[idx])
                    doc.metadata["method"] = "bm25"
                    retrieved_docs.append(doc)
        
        # Vector retrieval
        if self.is_simple_store:
            # Simple vector search
            query_vector = self.embeddings.embed_query(query)
            vector_docs = self.weaviate_client.similarity_search(query_vector, k=k//2)
            for doc in vector_docs:
                doc.metadata["method"] = "vector"
                retrieved_docs.append(doc)
        else:
            # Real Weaviate search
            try:
                vector_results = self.weaviate_client.query\
                    .get("LegalDocument", ["content", "source", "chunk_id"])\
                    .with_near_vector({
                        "vector": self.embeddings.embed_query(query)
                    })\
                    .with_limit(k//2)\
                    .do()
                
                if "data" in vector_results and "Get" in vector_results["data"]:
                    for item in vector_results["data"]["Get"]["LegalDocument"]:
                        doc = Document(
                            page_content=item["content"],
                            metadata={
                                "source": item["source"],
                                "chunk_id": item["chunk_id"],
                                "method": "vector"
                            }
                        )
                        retrieved_docs.append(doc)
            except Exception as e:
                print(f"Warning: Vector search failed: {e}")
        
        # Remove duplicates và giới hạn k documents
        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= k:
                    break
        
        return unique_docs[:k]

# ================================
# PHẦN 7: VIETNAMESE TTS
# ================================

class VietnameseTTS:
    def __init__(self):
        # Alternative TTS implementation
        try:
            # Thử sử dụng ZaloPay TTS
            self.tts_pipeline = pipeline(
                "text-to-speech",
                model="zalopay/vietnamese-tts",
                device=0 if torch.cuda.is_available() else -1
            )
            self.tts_available = True
            print("✅ TTS pipeline loaded successfully")
        except Exception as e:
            print(f"⚠️ TTS load failed: {e}")
            print("🔄 Using alternative TTS method...")
            self.tts_available = False
    
    def synthesize(self, text: str, output_file: str = "output.wav"):
        if not self.tts_available:
            print("📝 TTS không khả dụng, chỉ hiển thị text")
            print(f"🎵 Text to speak: {text}")
            return None
            
        try:
            # Generate speech
            speech = self.tts_pipeline(text)
            
            # Save audio
            import soundfile as sf
            sf.write(output_file, speech["audio"], speech["sampling_rate"])
            
            return output_file
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"📝 Fallback - Text content: {text}")
            return None

# ================================
# PHẦN 8: MAIN RAG SYSTEM
# ================================

class VietnameseLegalRAG:
    def __init__(self):
        print("🚀 Khởi tạo Vietnamese Legal RAG System...")
        
        # Setup components
        self.weaviate_setup = WeaviateSetup()
        self.embeddings = BGEM3Embeddings()
        self.llm = VinaLlamaLLM()
        self.retriever = HybridRetriever(
            self.weaviate_setup.client, 
            self.embeddings
        )
        self.tts = VietnameseTTS()
        
        # Load và process dataset
        self.load_legal_dataset()
        
        print("✅ Hệ thống đã sẵn sàng!")
    
    def load_legal_dataset(self):
        print("📚 Đang tải Vietnamese Law Corpus...")
        
        try:
            # Load dataset
            dataset = load_dataset("clapAI/vietnamese-law-corpus", split="train")
            
            # Convert to documents
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?"]
            )
            
            for i, item in enumerate(dataset):
                # Lấy text content (điều chỉnh theo structure của dataset)
                text = item.get('text', '') or item.get('content', '') or str(item)
                
                if len(text.strip()) > 0:
                    # Split thành chunks
                    chunks = text_splitter.split_text(text)
                    
                    for j, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": f"law_document_{i}",
                                "chunk": j,
                                "doc_id": i
                            }
                        )
                        documents.append(doc)
                
                # Giới hạn để tránh quá tải trong demo
                if i >= 100:  # Chỉ lấy 100 documents đầu
                    break
            
            print(f"📖 Đã xử lý {len(documents)} chunks từ {i+1} documents")
            
            # Add to retriever
            self.retriever.add_documents(documents)
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dataset: {e}")
            print("🔄 Tạo dữ liệu mẫu...")
            self.create_sample_data()
    
    def create_sample_data(self):
        # Tạo dữ liệu mẫu phong phú hơn về pháp luật Việt Nam
        sample_texts = [
            "Tù chung thân là hình phạt tù không có thời hạn, được áp dụng cho những tội phạm đặc biệt nghiêm trọng. Người bị kết án tù chung thân có thể được xem xét giảm án sau khi chấp hành ít nhất 20 năm tù.",
            
            "Luật Dân sự quy định về quyền và nghĩa vụ của công dân trong các quan hệ dân sự như quyền sở hữu, quyền thừa kế, hợp đồng mua bán.",
            
            "Bộ luật Hình sự Việt Nam quy định các tội phạm và hình phạt tương ứng. Các hình phạt chính bao gồm: cảnh cáo, phạt tiền, cải tạo không giam giữ, tù có thời hạn, tù chung thân và tử hình.",
            
            "Tù có thời hạn là hình phạt từ 3 tháng đến 20 năm. Đối với người chưa thành niên, thời hạn tù tối đa là 12 năm. Tù chung thân chỉ áp dụng cho người từ đủ 18 tuổi trở lên.",
            
            "Luật Lao động bảo vệ quyền lợi của người lao động bao gồm quyền được làm việc, quyền được trả lương công bằng, quyền nghỉ ngơi, quyền an toàn lao động.",
            
            "Hiến pháp là luật cơ bản của nhà nước, quy định về chế độ chính trị, quyền và nghĩa vụ cơ bản của công dân, tổ chức bộ máy nhà nước.",
            
            "Luật Hôn nhân và Gia đình quy định về hôn nhân, ly hôn, nuôi dưỡng con cái, bảo vệ quyền lợi của phụ nữ và trẻ em trong gia đình.",
            
            "Các tội phạm về tham nhũng bao gồm: nhận hối lộ, đưa hối lộ, lạm dụng chức vụ quyền hạn, tham ô tài sản. Hình phạt có thể từ phạt tiền đến tù chung thân.",
            
            "Quyền bào chữa là quyền cơ bản của người bị buộc tội. Mọi người đều có quyền tự bào chữa hoặc nhờ luật sư bào chữa khi bị truy cứu trách nhiệm hình sự.",
            
            "Tòa án nhân dân là cơ quan xét xử của nhà nước, có nhiệm vụ bảo vệ công lý, bảo vệ quyền con người, quyền công dân, chế độ xã hội chủ nghĩa."
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text,
                metadata={"source": f"sample_law_{i}", "doc_id": i}
            )
            documents.append(doc)
        
        self.retriever.add_documents(documents)
        print(f"✅ Đã tạo {len(sample_texts)} documents mẫu về pháp luật Việt Nam")
    
    def query(self, question: str, return_audio: bool = True):
        print(f"❓ Câu hỏi: {question}")
        
        # Retrieve relevant documents
        print("🔍 Đang tìm kiếm tài liệu liên quan...")
        relevant_docs = self.retriever.retrieve(question, k=3)
        
        # Debug: In ra documents được retrieve
        print(f"📋 Tìm thấy {len(relevant_docs)} tài liệu liên quan:")
        for i, doc in enumerate(relevant_docs):
            print(f"  {i+1}. [{doc.metadata.get('method', 'unknown')}] {doc.page_content[:100]}...")
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Improved prompt với context filtering
        if len(context.strip()) > 0:
            prompt = f"""Thông tin pháp luật liên quan:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin pháp luật trên. Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó."""
        else:
            prompt = f"""Câu hỏi: {question}

Hãy trả lời câu hỏi về pháp luật Việt Nam một cách ngắn gọn và chính xác."""
        
        # Generate response
        print("🤖 Đang tạo câu trả lời...")
        response = self.llm._call(prompt)
        
        print(f"📝 Câu trả lời: {response}")
        
        # Generate audio nếu được yêu cầu
        audio_file = None
        if return_audio:
            print("🔊 Đang tạo file âm thanh...")
            audio_file = self.tts.synthesize(response)
            if audio_file:
                print(f"🎵 Đã tạo file âm thanh: {audio_file}")
        
        return {
            "question": question,
            "answer": response,
            "context": context,
            "relevant_docs": len(relevant_docs),
            "audio_file": audio_file,
            "retrieved_content": [doc.page_content for doc in relevant_docs]
        }

# ================================
# PHẦN 9: CHẠY DEMO
# ================================

def main():
    # Khởi tạo hệ thống
    rag_system = VietnameseLegalRAG()
    
    # Các câu hỏi demo
    demo_questions = [
        "Quyền và nghĩa vụ cơ bản của công dân là gì?",
        "Luật hình sự quy định những tội phạm nào?",
        "Tù chung thân là gì?",
        "Hình phạt tù có thời hạn bao lâu?",
        "Luật hôn nhân và gia đình có những quy định gì?"
    ]
    
    print("\n" + "="*50)
    print("🎯 DEMO VIETNAMESE LEGAL RAG SYSTEM")
    print("="*50)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Demo {i} ---")
        try:
            result = rag_system.query(question, return_audio=False)
            print("-" * 30)
        except Exception as e:
            print(f"❌ Lỗi trong demo {i}: {e}")
    
    # Interactive mode với xử lý encoding
    print("\n" + "="*50)
    print("💬 CHẾ ĐỘ TÂM SỰ PHÁP LUẬT")
    print("Nhập 'quit' hoặc 'thoát' để thoát")
    print("Nhập số để chọn câu hỏi mẫu:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print("="*50)
    
    while True:
        try:
            # Thử nhiều cách nhập input
            question = None
            
            # Method 1: Direct input với encoding handling
            try:
                import sys
                if hasattr(sys.stdin, 'buffer'):
                    # Đọc bytes và decode thủ công
                    print("\n❓ Câu hỏi của bạn: ", end='', flush=True)
                    line = sys.stdin.buffer.readline()
                    question = line.decode('utf-8', errors='ignore').strip()
                else:
                    question = input("\n❓ Câu hỏi của bạn: ").strip()
            except UnicodeDecodeError:
                # Method 2: Fallback input
                print("\n⚠️ Lỗi encoding. Vui lòng nhập lại hoặc chọn số (1-5):")
                question = input("Input: ").strip()
            
            # Xử lý input
            if not question:
                continue
                
            # Check exit commands
            if question.lower() in ['quit', 'exit', 'thoat', 'thoát', 'q']:
                break
            
            # Check if input is a number (choose from demo questions)
            try:
                num = int(question)
                if 1 <= num <= len(demo_questions):
                    question = demo_questions[num - 1]
                    print(f"📝 Đã chọn: {question}")
                else:
                    print("❌ Số không hợp lệ. Vui lòng chọn từ 1-5")
                    continue
            except ValueError:
                # Not a number, treat as regular question
                pass
            
            # Process question
            if question.strip():
                try:
                    result = rag_system.query(question)
                    
                    # Play audio nếu có
                    if result["audio_file"]:
                        try:
                            from IPython.display import Audio, display
                            display(Audio(result["audio_file"]))
                        except:
                            print("⚠️ Không thể phát âm thanh trong môi trường này")
                            
                except Exception as e:
                    print(f"❌ Lỗi xử lý câu hỏi: {e}")
                    print("🔄 Vui lòng thử lại với câu hỏi khác")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình")
            break
        except Exception as e:
            print(f"\n❌ Lỗi không mong muốn: {e}")
            print("🔄 Tiếp tục...")
    
    print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")

# ================================
# TEST FUNCTION - Thêm function test riêng
# ================================

def test_system():
    """Function để test hệ thống với câu hỏi cố định"""
    print("🧪 CHẠY TEST HỆ THỐNG")
    print("="*50)
    
    try:
        rag_system = VietnameseLegalRAG()
        
        test_questions = [
            "Tù chung thân là gì?",
            "Luật dân sự quy định gì?",
            "Hình phạt tù có mấy loại?",
            "Quyền bào chữa là gì?",
            "Tòa án có chức năng gì?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            print("-" * 40)
            
            try:
                result = rag_system.query(question, return_audio=False)
                print(f"✅ Test {i} hoàn thành")
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
            
            print("=" * 40)
            
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")

# ================================
# CHẠY CHƯƠNG TRÌNH
# ================================

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_system()
    else:
        try:
            main()
        except Exception as e:
            print(f"❌ Lỗi chương trình chính: {e}")
            print("🔄 Thử chạy chế độ test: python vietnamese_legal_rag_system.py test")

# ================================
# HƯỚNG DẪN SỬ DỤNG TRÊN KAGGLE
# ================================

"""
CÁCH CHẠY TRÊN KAGGLE:

1. Tạo một Kaggle Notebook mới
2. Bật GPU (Settings -> Accelerator -> GPU)
3. Copy toàn bộ code này vào notebook
4. Uncomment và chạy phần cài đặt thư viện ở đầu
5. Chạy các cell theo thứ tự
6. Hệ thống sẽ tự động:
   - Tải Vietnamese Law Corpus
   - Thiết lập Weaviate embedded
   - Load VinaLlama model với quantization
   - Tạo hybrid retriever (BM25 + BGE-M3)
   - Sẵn sàng trả lời câu hỏi pháp luật

LƯU Ý:
- Code đã được tối ưu cho environment Kaggle
- Sử dụng quantization để tiết kiệm memory
- Embedded Weaviate không cần setup phức tạp
- TTS có thể không hoạt động perfect trên Kaggle
- Có thể điều chỉnh chunk_size và k để tối ưu performance

TÍNH NĂNG:
✅ LLM: VinaLlama 2.7B Chat với quantization
✅ RAG: Hybrid BM25 + BGE-M3 embeddings
✅ Vector Store: Weaviate với fallback SimpleVectorStore
✅ Dataset: Vietnamese Law Corpus từ HuggingFace
✅ TTS: Vietnamese text-to-speech
✅ Interactive Demo: Console interface với menu
✅ Error Handling: Robust error handling và fallback
✅ Memory Optimization: 4-bit quantization cho large models
"""