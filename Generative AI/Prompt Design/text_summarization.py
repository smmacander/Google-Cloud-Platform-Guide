#Install Vertex AI SDK
#!pip install google-cloud-aiplatform --upgrade --user

#Import libraries
from vertexai.language_models import TextGenerationModel

#Import models
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Transcript summarization
prompt = """
Provide a very short summary, no more than three sentences, for the following article:

Our quantum computers work by manipulating qubits in an orchestrated fashion that we call quantum algorithms.
The challenge is that qubits are so sensitive that even stray light can cause calculation errors — and the problem worsens as quantum computers grow.
This has significant consequences, since the best quantum algorithms that we know for running useful applications require the error rates of our qubits to be far lower than we have today.
To bridge this gap, we will need quantum error correction.
Quantum error correction protects information by encoding it across multiple physical qubits to form a “logical qubit,” and is believed to be the only way to produce a large-scale quantum computer with error rates low enough for useful calculations.
Instead of computing on the individual qubits themselves, we will then compute on logical qubits. By encoding larger numbers of physical qubits on our quantum processor into one logical qubit, we hope to reduce the error rates to enable useful quantum algorithms.

Summary:

"""

print(
    generation_model.predict(
        prompt, temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8
    ).text
)

prompt = """
Provide a TL;DR for the following article:

Our quantum computers work by manipulating qubits in an orchestrated fashion that we call quantum algorithms. 
The challenge is that qubits are so sensitive that even stray light can cause calculation errors — and the problem worsens as quantum computers grow. 
This has significant consequences, since the best quantum algorithms that we know for running useful applications require the error rates of our qubits to be far lower than we have today. 
To bridge this gap, we will need quantum error correction. 
Quantum error correction protects information by encoding it across multiple physical qubits to form a “logical qubit,” and is believed to be the only way to produce a large-scale quantum computer with error rates low enough for useful calculations. 
Instead of computing on the individual qubits themselves, we will then compute on logical qubits. By encoding larger numbers of physical qubits on our quantum processor into one logical qubit, we hope to reduce the error rates to enable useful quantum algorithms.

TL;DR:
"""

print(
    generation_model.predict(
        prompt, temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8
    ).text
)

#Summarize text into bullet points
prompt = """
Provide a very short summary in four bullet points for the following article:

Our quantum computers work by manipulating qubits in an orchestrated fashion that we call quantum algorithms.
The challenge is that qubits are so sensitive that even stray light can cause calculation errors — and the problem worsens as quantum computers grow.
This has significant consequences, since the best quantum algorithms that we know for running useful applications require the error rates of our qubits to be far lower than we have today.
To bridge this gap, we will need quantum error correction.
Quantum error correction protects information by encoding it across multiple physical qubits to form a “logical qubit,” and is believed to be the only way to produce a large-scale quantum computer with error rates low enough for useful calculations.
Instead of computing on the individual qubits themselves, we will then compute on logical qubits. By encoding larger numbers of physical qubits on our quantum processor into one logical qubit, we hope to reduce the error rates to enable useful quantum algorithms.

Bulletpoints:

"""

print(
    generation_model.predict(
        prompt, temperature=0.2, max_output_tokens=256, top_k=1, top_p=0.8
    ).text
)

#Dialogue summarization with to-dos
prompt = """
Please generate a summary of the following conversation and at the end summarize the to-do's for the support Agent:

Customer: Hi, I'm Larry, and I received the wrong item.

Support Agent: Hi, Larry. How would you like to see this resolved?

Customer: That's alright. I want to return the item and get a refund, please.

Support Agent: Of course. I can process the refund for you now. Can I have your order number, please?

Customer: It's [ORDER NUMBER].

Support Agent: Thank you. I've processed the refund, and you will receive your money back within 14 days.

Customer: Thank you very much.

Support Agent: You're welcome, Larry. Have a good day!

Summary:
"""

print(
    generation_model.predict(
        prompt, temperature=0.2, max_output_tokens=256, top_k=40, top_p=0.8
    ).text
)

#Hashtag tokenization
prompt = """
Tokenize the hashtags of this tweet:

Google Cloud
@googlecloud
How can data help our changing planet? 🌎

In honor of #EarthDay this weekend, we’re proud to share how we’re partnering with
@ClimateEngine
 to harness the power of geospatial data and drive toward a more sustainable future.

Check out how → https://goo.gle/3mOUfts
"""

print(
    generation_model.predict(
        prompt, temperature=0.8, max_output_tokens=1024, top_k=40, top_p=0.8
    ).text
)

#Title & heading generation
prompt = """
Write a title for this text, give me five options:
Whether helping physicians identify disease or finding photos of “hugs,” AI is behind a lot of the work we do at Google. And at our Arts & Culture Lab in Paris, we’ve been experimenting with how AI can be used for the benefit of culture.
Today, we’re sharing our latest experiments—prototypes that build on seven years of work in partnership the 1,500 cultural institutions around the world.
Each of these experimental applications runs AI algorithms in the background to let you unearth cultural connections hidden in archives—and even find artworks that match your home decor."
"""

print(
    generation_model.predict(
        prompt, temperature=0.8, max_output_tokens=256, top_k=1, top_p=0.8
    ).text
)

#Evaluation
#!pip install rouge

from rouge import Rouge

ROUGE = Rouge()

prompt = """
Provide a very short, maximum four sentences, summary for the following article:

Our quantum computers work by manipulating qubits in an orchestrated fashion that we call quantum algorithms.
The challenge is that qubits are so sensitive that even stray light can cause calculation errors — and the problem worsens as quantum computers grow.
This has significant consequences, since the best quantum algorithms that we know for running useful applications require the error rates of our qubits to be far lower than we have today.
To bridge this gap, we will need quantum error correction.
Quantum error correction protects information by encoding it across multiple physical qubits to form a “logical qubit,” and is believed to be the only way to produce a large-scale quantum computer with error rates low enough for useful calculations.
Instead of computing on the individual qubits themselves, we will then compute on logical qubits. By encoding larger numbers of physical qubits on our quantum processor into one logical qubit, we hope to reduce the error rates to enable useful quantum algorithms.

Summary:

"""

candidate = generation_model.predict(
    prompt, temperature=0.1, max_output_tokens=1024, top_k=40, top_p=0.9
).text

print(candidate)

reference = "Quantum computers are sensitive to noise and errors. To bridge this gap, we will need quantum error correction. Quantum error correction protects information by encoding across multiple physical qubits to form a “logical qubit”."

ROUGE.get_scores(candidate, reference)