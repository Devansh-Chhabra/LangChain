from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import load_prompt

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    device = -1, # force to use CPU
    pipeline_kwargs = dict(
        temperature = 1.5
    )
)
model = ChatHuggingFace(llm = llm)

paper_input = "Attention Is All You Need"
style_input = "Beginner-Friendly"
length_input = "Long (detailed explanation)"

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

result = model.invoke(prompt)
print(result.content)