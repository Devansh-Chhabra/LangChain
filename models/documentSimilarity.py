from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query = "facts about bumrah"

doc_embeddings = embedding.embed_documents(documents)
q_embedding = embedding.embed_query(query)

scores = cosine_similarity([q_embedding],doc_embeddings)[0]
index,score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1] # max similarity score will be present at last along with correct index

print(query)
print(documents[index])
print("similarity score is:", score)