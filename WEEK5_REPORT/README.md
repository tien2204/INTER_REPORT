# 🎤 Voice Agent System

Hệ thống Voice Agent cho phép người dùng tương tác bằng giọng nói để truy vấn thông tin từ các tài liệu văn bản và hình ảnh. Sử dụng công nghệ RAG (Retrieval-Augmented Generation) và các mô hình AI tiên tiến để xử lý và trả lời câu hỏi một cách tự nhiên.

## 🧱 Kiến trúc hệ thống

### 1. Cấu trúc module
```
voice_agent/
├── app/                     # Điểm vào của ứng dụng
│   └── main.py             # File chính chạy ứng dụng
├── config/                 # Cấu hình hệ thống
│   ├── settings.py        # Cài đặt hệ thống
│   └── logging_config.py  # Cấu hình logging
├── core/                  # Các module lõi
│   ├── voice/            # Xử lý giọng nói
│   │   ├── stt.py       # Speech-to-Text
│   │   └── tts.py       # Text-to-Speech
│   ├── text/             # Xử lý văn bản
│   │   ├── question_rewriter.py # Rewrite câu hỏi
│   │   └── reranker.py   # Reranking tài liệu
│   └── graph/            # Graph-based RAG
│       └── graphrag.py   # Hệ thống RAG dựa trên đồ thị
├── services/             # Dịch vụ xử lý
│   ├── document/        # Xử lý tài liệu
│   │   ├── processor.py # Xử lý tài liệu
│   │   └── extractor.py # Trích xuất văn bản
│   └── image/           # Xử lý hình ảnh
│       └── processor.py # OCR và xử lý hình ảnh
├── utils/               # Các công cụ hỗ trợ
│   └── history_manager.py # Quản lý lịch sử cuộc trò chuyện (tự động tạo và quản lý trong thư mục data/history)
└── requirements.txt     # Yêu cầu phần mềm
```

### 2. Workflow hệ thống
1. **Nhận dạng giọng nói**
   - STT: Nhận dạng giọng nói thành văn bản
   - Wake word detection: Phát hiện từ kích hoạt

2. **Xử lý câu hỏi**
   - Rewrite câu hỏi: Tạo nhiều phiên bản câu hỏi để cải thiện tìm kiếm context
   - Phân tích ngữ cảnh: Sử dụng lịch sử cuộc trò chuyện làm context

3. **Truy vấn dữ liệu**
   - OCR: Xử lý hình ảnh và PDF chứa văn bản
   - Vector search: Tìm kiếm dựa trên vector embedding
   - GraphRAG: Sử dụng đồ thị để xử lý câu hỏi phức tạp

4. **Tạo câu trả lời**
   - RAG: Kết hợp thông tin từ tài liệu và LLM
   - Reranking: Sắp xếp lại kết quả theo độ liên quan
   - TTS: Chuyển đổi văn bản thành giọng nói

## 📦 Yêu cầu hệ thống

### Phần mềm
- Python >= 3.10
- Các thư viện chính:
  - `langchain` cho RAG
  - `transformers` cho xử lý ngôn ngữ
  - `sentence-transformers` cho embedding
  - `PyPDF2` và `python-docx` cho xử lý tài liệu
  - `pytesseract` cho OCR
  - `speech_recognition` cho STT
  - `pyttsx3` cho TTS

### Phần cứng
- CPU: Intel Core i5/i7 hoặc tương đương
- RAM: 8GB trở lên
- Ổ cứng: SSD để tăng tốc độ xử lý
- Microphone và loa chất lượng tốt

## 🚀 Cách cài đặt và chạy

1. **Cài đặt môi trường**
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt các thư viện
pip install -r requirements.txt
```

2. **Cấu hình môi trường**
- Sao chép file `.env.example` thành `.env`
- Cập nhật các biến môi trường:
  - OPENAI_API_KEY
  - SAMPLE_RATE
  - CHUNK_SIZE
  - CHUNK_OVERLAP

3. **Chạy ứng dụng**
```bash
# Chạy ứng dụng
python app/main.py
```

## 🧠 Giải quyết các vấn đề

### 1. Rewrite câu hỏi
- Sử dụng GPT-3.5-turbo để tạo nhiều phiên bản câu hỏi
- Giúp cải thiện việc tìm kiếm context từ tài liệu
- Tăng cơ hội tìm thấy thông tin liên quan

### 2. ChromaDB và Vector Indexing
- Sử dụng HNSW indexing của ChromaDB
- Tối ưu hóa việc tìm kiếm vector tương đồng
- Giảm thời gian truy vấn và tăng hiệu suất

### 3. Reranking và Tradeoff
- Sử dụng hai mô hình chuyên biệt cho reranking:
  - `all-MiniLM-L6-v2`: Tạo embeddings ban đầu
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`: Đánh giá độ liên quan chính xác
- Quy trình reranking 2 bước:
  1. Sử dụng embeddings để lọc top 20 tài liệu
  2. Sử dụng cross-encoder để đánh giá chính xác
- Đảm bảo hiệu suất trên CPU
- Cân bằng giữa tốc độ và chất lượng kết quả
- Phù hợp cho Voice Agent cần phản hồi nhanh

### 4. Xử lý hình ảnh và OCR
- Sử dụng pytesseract cho OCR
- Hỗ trợ xử lý hình ảnh và PDF
- Tích hợp Vision LLM cho hình ảnh phức tạp
- Tự động trích xuất văn bản từ hình ảnh

### 5. Quản lý lịch sử cuộc trò chuyện
- Hệ thống tự động lưu trữ lịch sử cuộc trò chuyện trong thư mục data/history
- Sử dụng làm context cho câu hỏi tiếp theo
- Giúp duy trì mạch lạc cuộc trò chuyện
- Tự động tải lịch sử gần đây khi khởi động

### 6. GraphRAG cho câu hỏi phức tạp
- Sử dụng đồ thị để kết nối thông tin liên quan
- Xử lý câu hỏi tổng hợp và phức tạp
- Tạo ngữ cảnh tốt hơn cho câu trả lời
- Tăng độ chính xác cho câu hỏi phức tạp

## 🛠️ Tùy chỉnh nâng cao

1. **Điều chỉnh cấu hình**
- Thay đổi kích thước chunk văn bản
- Điều chỉnh ngưỡng tương đồng
- Tùy chỉnh tốc độ nói
- Thay đổi giọng nói

2. **Tối ưu hóa hiệu suất**
- Điều chỉnh batch size
- Sử dụng CPU/GPU tùy chọn
- Tối ưu hóa bộ nhớ
- Cập nhật mô hình mới

## 📁 Cấu trúc thư mục chi tiết

```
voice_agent/
├── app/                     # Điểm vào của ứng dụng
├── config/                 # Cấu hình
├── core/                  # Module lõi
├── services/             # Dịch vụ xử lý
├── utils/               # Công cụ hỗ trợ
├── requirements.txt     # Yêu cầu phần mềm
├── .env.example         # Mẫu file cấu hình
├── README.md           # Tài liệu hướng dẫn
└── data/               # Thư mục dữ liệu
    ├── input/         # Người dùng thêm file tài liệu vào đây
    ├── embeddings/    # Hệ thống tự tạo và quản lý
    └── history/       # Hệ thống tự tạo và quản lý
```

### Lưu ý về thư mục dữ liệu:
- `data/input/`: Người dùng cần tạo thư mục này và thêm các file tài liệu cần xử lý (PDF, Word, Excel, hình ảnh)
- `data/embeddings/` và `data/history/`: Hai thư mục này sẽ được hệ thống tự tạo và quản lý trong quá trình chạy
  - `embeddings/`: Lưu trữ vector embeddings của tài liệu
  - `history/`: Lưu trữ lịch sử cuộc trò chuyện

## 📝 Lưu ý quan trọng

1. **Hiệu suất**
- Hệ thống được tối ưu hóa cho CPU
- Sử dụng MiniLM cho reranking
- Có thể điều chỉnh batch size để tối ưu

2. **Bảo mật**
- Không lưu trữ thông tin nhạy cảm
- Sử dụng biến môi trường cho API keys
- Mã hóa dữ liệu nhạy cảm

3. **Đóng góp**
- Báo cáo lỗi qua issue tracker
- Gửi pull request cho cải tiến
- Tham gia thảo luận trên forum

## 📚 Tài liệu tham khảo

- [Understanding Vector Indexing](https://medium.com/@myscale/understanding-vector-indexing-a-comprehensive-guide-d1abe36ccd3c)
- [LangChain Documentation](https://docs.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
