# Hệ thống Hỏi Đáp bằng Giọng Nói từ Tài Liệu

Hệ thống cho phép bạn hỏi đáp về tài liệu (PDF, Word, Excel) bằng giọng nói và nhận câu trả lời bằng giọng nói.

## Các Thành Phần Chính

1. **Trích xuất dữ liệu từ tài liệu văn bản**
   - Sử dụng PyPDF2 để xử lý PDF
   - Sử dụng python-docx để xử lý Word
   - Sử dụng pandas để xử lý Excel
   - Sử dụng langchain.text_splitter để tách văn bản thành chunks

2. **Hệ thống Hỏi Đáp từ Tài Liệu (RAG)**
   - Sử dụng OpenAIEmbeddings để tạo vector
   - Sử dụng ChromaDB để lưu trữ vector
   - Sử dụng ChatOpenAI để xử lý ngôn ngữ tự nhiên
   - Sử dụng RetrievalQA để tạo hệ thống Q&A

3. **Chuyển đổi Giọng Nói thành Văn Bản (STT)**
   - Sử dụng Whisper của OpenAI
   - Sử dụng faster-whisper để tăng tốc độ
   - Có thể fallback sang Google Speech Recognition

4. **Chuyển đổi Văn Bản thành Giọng Nói (TTS)**
   - Sử dụng pyttsx3 (offline và nhanh)
   - Hỗ trợ đa ngôn ngữ
   - Có thể tùy chỉnh tốc độ và âm lượng

## Luồng Hoạt Động

1. **Khởi động Hệ thống**
   ```python
   def main():
       # 1. Xử lý tài liệu
       document_processor.process_documents()
       
       # 2. Khởi động các component
       speech_processor = SpeechProcessor()
       tts_processor = TTSProcessor()
       rag_processor = RAGProcessor()
   ```

2. **Chu trình Làm Việc**
   ```python
   while True:
       # 1. Nghe wake word
       wake_word = speech_processor.listen_for_wake_word()
       
       if wake_word:
           # 2. Trả lời khi kích hoạt
           tts_processor.speak("I'm here. How can I help you?")
           
           # 3. Nghe câu hỏi
           query = speech_processor.listen_for_query()
           
           if query:
               # 4. Trả lời câu hỏi
               tts_processor.speak("I'm on it!")
               
               # 5. Xử lý RAG
               response = rag_processor.process_query(query)
               
               # 6. Đọc câu trả lời
               tts_processor.speak(response)
   ```

## Cách Sử Dụng

1. **Cài đặt**
```bash
pip install -r requirements.txt
```

2. **Cấu hình**
   - Cập nhật OpenAI API key trong file .env
   - Tạo thư mục `documents` để chứa tài liệu
   - Đảm bảo tai nghe Bluetooth được kết nối

3. **Chạy hệ thống**
```bash
python main.py
```

4. **Sử dụng**
   - Nói "Hey Assistant" để kích hoạt
   - Đặt câu hỏi về tài liệu
   - Nghe câu trả lời
   - Nói "Stop listening" để kết thúc

## Ví dụ Sử Dụng

```python
# Người dùng nói: "Hey Assistant"
# -> Hệ thống kích hoạt và trả lời: "I'm here. How can I help you?"

# Người dùng hỏi: "What's the total sales in Q1?"
# -> Hệ thống:
1. Chuyển đổi speech thành text
2. Tìm kiếm thông tin từ tài liệu Excel
3. Tính toán và trả lời
4. Đọc câu trả lời: "The total sales in Q1 was $100,000"
```

## Yêu cầu Hệ thống

- Python 3.8+
- OpenAI API key
- Tai nghe Bluetooth
- Định dạng tài liệu hỗ trợ:
  - PDF (.pdf)
  - Word (.doc, .docx)
  - Excel (.xls, .xlsx)

## Khắc Phục Lỗi

1. Nếu không nhận diện được giọng nói:
   - Kiểm tra tai nghe đã kết nối chưa
   - Kiểm tra âm lượng micro
   - Thử nói rõ ràng hơn

2. Nếu không có âm thanh:
   - Kiểm tra âm lượng loa
   - Kiểm tra kết nối tai nghe
   - Thử khởi động lại ứng dụng

3. Nếu không xử lý được tài liệu:
   - Kiểm tra định dạng file
   - Kiểm tra quyền truy cập file
   - Kiểm tra ChromaDB có hoạt động không
