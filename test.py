# Cài đặt thư viện: pip install unstructured langchain-text-splitters

from langchain_text_splitters import MarkdownHeaderTextSplitter

# Giả sử bạn đã convert tài liệu sang Markdown (đây là cách tốt nhất cho LightRAG)
# Cấu trúc tài liệu giả lập:
markdown_document = """
# Giới thiệu về AI
AI là một lĩnh vực rộng lớn của khoa học máy tính.

## Machine Learning
Machine Learning là một tập con của AI.
Nó tập trung vào việc học từ dữ liệu.

### Deep Learning
Deep Learning sử dụng mạng thần kinh nhân tạo.

## Vision-Language Models (VLM)
Đây là lĩnh vực kết hợp giữa hình ảnh và ngôn ngữ.
"""

# Bước 1: Định nghĩa các cấp độ cấu trúc muốn cắt
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Bước 2: Khởi tạo bộ cắt dựa trên Header
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False # Giữ lại header trong nội dung chunk
)

# Bước 3: Thực hiện phân đoạn
md_header_splits = markdown_splitter.split_text(markdown_document)

# Bước 4: Hiển thị kết quả thực nghiệm
for i, chunk in enumerate(md_header_splits):
    print(f"--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}") # Chứa thông tin cấu trúc (Header cha)
    print("\n")