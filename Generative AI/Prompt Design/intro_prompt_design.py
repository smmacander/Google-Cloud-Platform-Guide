#Install Vertex AI SDK
#!pip install "shapely<2.0.0"
#!pip install google-cloud-aiplatform --upgrade --user

#Import libraries
from vertexai.language_models import TextGenerationModel

#Load model
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Be concise
#Not recommended
prompt = "What do you think could be a good name for a flower shop that specializes in selling bouquets of dried flowers more than fresh flowers? Thank you!"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Recommended
prompt = "Suggest a name for a flower shop that sells bouquets of dried flowers"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Be specific, and well-defined
#Not recommended
prompt = "Tell me about Earth"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Recommended
prompt = "Generate a list of ways that makes Earth unique compared to other planets"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Ask one task at a time
#Not recommended
prompt = "What's the best method of boiling water and why is the sky blue?"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Recommended
prompt = "What's the best method of boiling water?"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)
prompt = "Why is the sky blue?"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Watch out for hallucinations
prompt = "Who was the first elephant to visit the moon?"

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Turn generative tasks into classification tasks to reduce output variability
prompt = "I'm a high school student. Recommend me a programming activity to improve my skills."

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Classification tasks reduces output variability
prompt = """I'm a high school student. Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Improve response quality by invluding examples
#Zero-shot prompt
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment:
"""

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#One-shot prompt
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment:
"""

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)

#Few-shot prompt
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment: negative

Tweet: Something surprised me about this video - it was actually original. It was not the same old recycled stuff that I always see. Watch it - you will not regret it.
Sentiment:
"""

print(generation_model.predict(prompt=prompt, max_output_tokens=256).text)