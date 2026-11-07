from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)