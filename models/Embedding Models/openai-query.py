from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32) # default dimension of embedding is 3072

text = "Delhi is the capital of India"
result = embedding.embed_query(text)
print(str(result))