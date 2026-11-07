from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    device = -1, # force to use CPU
    pipeline_kwargs = dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

# 1) create the template
template = PromptTemplate(
    template = "Greet this person in 5 languages. The name of the person is {name}",
    input_variables = ['name']
)

# 2) fill values of placeholders
prompt = template.invoke({'name':'Devansh'})

# get the result
result = model.invoke(prompt)
print(result.content)