#Install Vertex AI SDK
#!pip install google-cloud-aiplatform --upgrade --user

#Import Libraries
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextGenerationModel, \
                                     TextEmbeddingModel, \
                                     ChatModel, \
                                     InputOutputTextPair, \
                                     CodeGenerationModel, \
                                     CodeChatModel

#Load model
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Hello PaLM
prompt = "What is a large language model?"

response = generation_model.predict(prompt=prompt)

print(response.text)

#Try out your own prompt
prompt = """Create a numbered list of 10 items. Each item in the list should be a trend in the tech industry.

Each trend should be less than 5 words.""" # try your own prompt

response = generation_model.predict(prompt=prompt)

print(response.text)

#Prompt templates
my_industry = "tech" # try changing this to a different industry

response = generation_model.predict(
    prompt=f"""Create a numbered list of 10 items. Each item in the list should
    be a trend in the {my_industry} industry.

    Each trend should be less than 5 words."""
)

print(response.text)

#The temperature parameter (range: 0.0 - 1.0, default 0)
temp_val = 0.0
prompt_temperature = "Complete the sentence: As I prepared the picture frame, I reached into my toolkit to fetch my:"

response = generation_model.predict(
    prompt=prompt_temperature,
    temperature=temp_val,
)

print(f"[temperature = {temp_val}]")
print(response.text)

temp_val = 1.0

response = generation_model.predict(
    prompt=prompt_temperature,
    temperature=temp_val,
)

print(f"[temperature = {temp_val}]")
print(response.text)

#The max_output_tokens parameter (range: 1 - 1024, default 128)
max_output_tokens_val = 5

response = generation_model.predict(
    prompt="List ten ways that generative AI can help improve the online shopping experience for users",
    max_output_tokens=max_output_tokens_val,
)

print(f"[max_output_tokens = {max_output_tokens_val}]")
print(response.text)

max_output_tokens_val = 500

response = generation_model.predict(
    prompt="List ten ways that generative AI can help improve the online shopping experience for users",
    max_output_tokens=max_output_tokens_val,
)

print(f"[max_output_tokens = {max_output_tokens_val}]")
print(response.text)

display(Markdown(response.text))

#The top_p parameter (range: 0.0 - 1.0, default 0.95)
top_p_val = 0.0
prompt_top_p_example = (
    "Create a marketing campaign for jackets that involves blue elephants and avocados."
)

response = generation_model.predict(
    prompt=prompt_top_p_example, temperature=0.9, top_p=top_p_val
)

print(f"[top_p = {top_p_val}]")
print(response.text)

top_p_val = 1.0

response = generation_model.predict(
    prompt=prompt_top_p_example, temperature=0.9, top_p=top_p_val
)

print(f"[top_p = {top_p_val}]")
print(response.text)

#The top_k parameter (range: 0.0 - 40, default 40)
prompt_top_k_example = "Write a 2-day itinerary for France."
top_k_val = 1

response = generation_model.predict(
    prompt=prompt_top_k_example, max_output_tokens=300, temperature=0.9, top_k=top_k_val
)

print(f"[top_k = {top_k_val}]")
print(response.text)

top_k_val = 40

response = generation_model.predict(
    prompt=prompt_top_k_example,
    max_output_tokens=300,
    temperature=0.9,
    top_k=top_k_val,
)

print(f"[top_k = {top_k_val}]")
print(response.text)

#Chat model with chat-bison@001
chat_model = ChatModel.from_pretrained("chat-bison@001")

chat = chat_model.start_chat()

print(
    chat.send_message(
        """
Hello! Can you write a 300 word abstract for a research paper I need to write about the impact of AI on society?
"""
    )
)

print(
    chat.send_message(
        """
Could you give me a catchy title for the paper?
"""
    )
)

#Advanced Chat model with the SDK
chat = chat_model.start_chat(
    context="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.",
    examples=[
        InputOutputTextPair(
            input_text="Who do you work for?",
            output_text="I work for Ned.",
        ),
        InputOutputTextPair(
            input_text="What do I like?",
            output_text="Ned likes watching movies.",
        ),
    ],
    temperature=0.3,
    max_output_tokens=200,
    top_p=0.8,
    top_k=40,
)
print(chat.send_message("Are my favorite movies based on a book series?"))

print(chat.send_message("When where these books published?"))

#Embedding model with textembedding-gecko@001
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

embeddings = embedding_model.get_embeddings(["What is life?"])

for embedding in embeddings:
    vector = embedding.values
    print(f"Length = {len(vector)}")
    print(vector)

#Embeddings and Pandas DataFrames
text = [
    "i really enjoyed the movie last night",
    "so many amazing cinematic scenes yesterday",
    "had a great time writing my Python scripts a few days ago",
    "huge sense of relief when my .py script finally ran without error",
    "O Romeo, Romeo, wherefore art thou Romeo?",
]

df = pd.DataFrame(text, columns=["text"])
df

df["embeddings"] = [
    emb.values for emb in embedding_model.get_embeddings(df.text.values)
]
df

#Comparing similarity of text examples using cosine similarity
cos_sim_array = cosine_similarity(list(df.embeddings.values))

# display as DataFrame
df = pd.DataFrame(cos_sim_array, index=text, columns=text)
df

ax = sns.heatmap(df, annot=True, cmap="crest")
ax.xaxis.tick_top()
ax.set_xticklabels(text, rotation=90)

#Code generation with code-bison@001
#Load model
code_generation_model = CodeGenerationModel.from_pretrained("code-bison@001")

#Hello Codey
prefix = "write a python function to do binary search"

response = code_generation_model.predict(prefix=prefix)

print(response.text)

#Try out your own prompt 
prefix = """write a python function named as "calculate_cosine_similairty" and three unit \
            tests where it takes two arguments "vector1" and "vector2". \
            It then uses numpy dot function to calculate the dot product of the two vectors. \n
          """

response = code_generation_model.predict(prefix=prefix, max_output_tokens=1024)

print(response.text)

#Prompt templates
language = "C++ function"
file_format = "json"
extract_info = "names"
requirments = """
              - the name should be start with capital letters.
              - There should be no duplicate names in the final list.
              """

prefix = f"""Create a {language} to parse {file_format} and extract {extract_info} with the following requirements: {requirments}.
              """

response = code_generation_model.predict(prefix=prefix, max_output_tokens=1024)

print(response.text)

#Code completion with code-gecko@001
code_completion_model = CodeGenerationModel.from_pretrained("code-gecko@001")

prefix = """
          def find_x_in_string(string_s, x):
         """

response = code_completion_model.predict(prefix=prefix,
                                         max_output_tokens=64)

print(response.text)

prefix = """
         def reverse_string(s):
            return s[::-1]
         def test_empty_input_string()
         """

response = code_completion_model.predict(prefix=prefix,
                                         max_output_tokens=64)

print(response.text)

#Code chat with codechat-bison@001
code_chat_model = CodeChatModel.from_pretrained("codechat-bison@001")

code_chat = code_chat_model.start_chat()

print(code_chat.send_message(
        "Please help write a function to calculate the min of two numbers",
    )
)

print(code_chat.send_message(
        "can you explain the code line by line in bullets?",
    )
)

code_chat = code_chat_model.start_chat()

print(code_chat.send_message(
        "what is the most scalable way to traverse a list in python?",
    )
)

print(code_chat.send_message(
        "how would i measure the iteration per second for the following code?",

    )
)