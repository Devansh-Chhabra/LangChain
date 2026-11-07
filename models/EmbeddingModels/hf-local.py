from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2') # dimension of embedding is 384(fixed)

text = "Delhi is the capital of India"
result = embedding.embed_query(text)
print(str(result))