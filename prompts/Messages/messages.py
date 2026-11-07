from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    device = -1, # force to use CPU
    pipeline_kwargs = dict(
        temperature = 0.1,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = 'You are a helpful assistant'),
    HumanMessage(content = 'Tell me about LangChain')
]
result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
print(messages)