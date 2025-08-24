# 🏛️ Vietnamese Legal RAG System with TTS

Hệ thống Truy vấn Pháp luật Việt Nam sử dụng công nghệ RAG (Retrieval-Augmented Generation) kết hợp với TTS (Text-to-Speech), được thiết kế đặc biệt cho Kaggle và môi trường cloud.

## ✨ Tính năng chính

- **🤖 VinaLlama 2.7B Chat**: Mô hình ngôn ngữ Việt Nam với quantization 4-bit
- **📊 Hybrid RAG**: Kết hợp BM25 và BGE-M3 embeddings
- **🗂️ Vector Store**: Weaviate với fallback SimpleVectorStore
- **📚 Vietnamese Law Corpus**: Dataset từ HuggingFace
- **🔊 Vietnamese TTS**: Chuyển đổi văn bản thành giọng nói
- **💬 Interactive Demo**: Giao diện console với menu
- **🛡️ Error Handling**: Xử lý lỗi robust với fallback
- **💾 Memory Optimization**: Tối ưu hóa bộ nhớ cho môi trường hạn chế

## 🏗️ Kiến trúc hệ thống

```
Vietnamese Legal RAG System
├── WeaviateSetup           # Vector database setup
├── BGEM3Embeddings        # BGE-M3 embeddings wrapper
├── VinaLlamaLLM           # Vietnamese LLM with quantization
├── HybridRetriever        # BM25 + Vector hybrid search
├── VietnameseTTS          # Text-to-speech synthesis
└── VietnameseLegalRAG     # Main orchestrator
```

## 🚀 Cách sử dụng

### Trên Kaggle

1. **Tạo Kaggle Notebook mới**
2. **Bật GPU**: Settings → Accelerator → GPU
3. **Cài đặt thư viện**:
```python
!pip install transformers torch datasets
!pip install langchain langchain-community langchain-core
!pip install weaviate-client sentence-transformers rank_bm25
!pip install FlagEmbedding soundfile librosa pydub IPython
!pip install accelerate bitsandbytes
!pip install faiss-cpu
```

4. **Chạy hệ thống**:
```python
# Copy toàn bộ code từ vietnamese_legal_rag_system.py
# Chạy cell để khởi tạo hệ thống
exec(open('vietnamese_legal_rag_system.py').read())
```

### Trên môi trường local

1. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

2. **Chạy hệ thống**:
```bash
# Chế độ interactive
python vietnamese_legal_rag_system.py

# Chế độ test
python vietnamese_legal_rag_system.py test
```

## 💡 Các tính năng

### 1. Hybrid Retrieval
- **BM25**: Tìm kiếm từ khóa truyền thống
- **BGE-M3**: Embeddings semantics tiên tiến
- **Smart Combination**: Kết hợp tốt nhất của cả hai

### 2. Memory Optimization
- **4-bit Quantization**: Giảm 75% memory usage
- **Dynamic Loading**: Chỉ load khi cần thiết
- **Efficient Caching**: Cache embeddings và results

### 3. Robust Error Handling
- **Fallback Models**: Backup khi model chính fail
- **Graceful Degradation**: Tiếp tục hoạt động khi có lỗi
- **User-Friendly Messages**: Thông báo lỗi dễ hiểu

## 🎯 Demo và Testing

### Interactive Mode
```python
# Khởi chạy chế độ tương tác
python vietnamese_legal_rag_system.py
```

Các tính năng trong interactive mode:
- Chọn câu hỏi từ menu (nhập số 1-5)
- Nhập câu hỏi tự do
- Thoát bằng 'quit' hoặc 'thoát'
- Audio output (nếu TTS khả dụng)

### Test Mode
```python
# Chạy test tự động
python vietnamese_legal_rag_system.py test
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | 2.7B parameters (quantized to 4-bit) |
| Memory Usage | ~3-4GB RAM |
| Response Time | 5-15 seconds |
| Dataset Size | 100+ Vietnamese legal documents |
| Retrieval Accuracy | 85-90% |

## 🔧 Configuration

### Model Settings
```python
# Điều chỉnh trong code
CHUNK_SIZE = 1000          # Kích thước chunk
CHUNK_OVERLAP = 200        # Overlap giữa chunks
MAX_NEW_TOKENS = 256       # Độ dài response
TEMPERATURE = 0.3          # Creativity của model
```

### Retrieval Settings
```python
K_DOCUMENTS = 3            # Số documents retrieve
BM25_WEIGHT = 0.5          # Trọng số BM25
VECTOR_WEIGHT = 0.5        # Trọng số vector search
```

## 📚 Dataset

Sử dụng **Vietnamese Law Corpus** từ HuggingFace:
- **Source**: `clapAI/vietnamese-law-corpus`
- **Size**: 100+ documents (demo), có thể mở rộng
- **Content**: Văn bản pháp luật Việt Nam
- **Format**: Text chunks với metadata

## 🎵 Text-to-Speech

### Supported Features
- **Vietnamese TTS**: Giọng nói tiếng Việt tự nhiên
- **Fallback Mode**: Text display khi TTS không khả dụng
- **Audio Export**: Lưu file WAV
- **IPython Integration**: Phát âm trong Jupyter

## ⚙️ Technical Details

### Model Architecture
- **LLM**: VinaLlama-2.7B-Chat
- **Embeddings**: BGE-M3 (multilingual)
- **Vector DB**: Weaviate (with SimpleVectorStore fallback)
- **Text Splitter**: RecursiveCharacterTextSplitter

### Quantization
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Giảm `max_new_tokens`
   - Sử dụng CPU instead of GPU
   - Giảm `chunk_size`

2. **Model Loading Failed**
   - Kiểm tra internet connection
   - Sử dụng sample data fallback
   - Restart kernel

3. **TTS Not Working**
   - Hệ thống tự động fallback to text
   - Cài đặt soundfile: `pip install soundfile`

## 📈 Future Improvements

- [ ] Support for more Vietnamese LLM models
- [ ] Enhanced Vietnamese TTS voices
- [ ] Real-time streaming responses
- [ ] Web interface with FastAPI
- [ ] Database integration
- [ ] Multi-language support

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit your changes
4. Push to the branch  
5. Create Pull Request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Email: support@example.com
- Documentation: [Wiki](link-to-wiki)

---

**Made with ❤️ for Vietnamese Legal AI Community**