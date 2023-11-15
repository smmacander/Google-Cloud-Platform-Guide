#Install Vertex AI SDK
#!pip install google-cloud-aiplatform --upgrade --user

#Import libraries
import pandas as pd
from vertexai.language_models import TextGenerationModel

#Import models
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Zero-shot prompting
prompt = """Q: Who was President of the United States in 1955? Which party did he belong to?\n
            A:
         """
print(
    generation_model.predict(
        prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

prompt = """Q: What is the tallest mountain in the world?\n
            A:
         """
print(
    generation_model.predict(
        prompt,
        max_output_tokens=20,
        temperature=0.1,
    ).text
)

#Few-shot prompting
prompt = """Q: Who is the current President of France?\n
            A: Emmanuel Macron \n\n

            Q: Who invented the telephone? \n
            A: Alexander Graham Bell \n\n

            Q: Who wrote the novel "1984"?
            A: George Orwell

            Q: Who discovered penicillin?
            A:
         """
print(
    generation_model.predict(
        prompt,
        max_output_tokens=20,
        temperature=0.1,
    ).text
)

#Adding internal knowledge as context in prompts
context = """
Storage and content policy \n
How durable is my data in Cloud Storage? \n
Cloud Storage is designed for 99.999999999% (11 9's) annual durability, which is appropriate for even primary storage and
business-critical applications. This high durability level is achieved through erasure coding that stores data pieces redundantly
across multiple devices located in multiple availability zones.
Objects written to Cloud Storage must be redundantly stored in at least two different availability zones before the
write is acknowledged as successful. Checksums are stored and regularly revalidated to proactively verify that the data
integrity of all data at rest as well as to detect corruption of data in transit. If required, corrections are automatically
made using redundant data. Customers can optionally enable object versioning to add protection against accidental deletion.
"""

question = "How is high availability achieved?"

prompt = f"""Answer the question given in the contex below:
Context: {context}?\n
Question: {question} \n
Answer:
"""

print("[Prompt]")
print(prompt)

print("[Response]")
print(
    generation_model.predict(
        prompt,
    ).text
)

#Instruction-tuning outputs
question = "What machined are required for hosting Vertex AI models?"
prompt = f"""Answer the question given the context below as {{Context:}}. \n
If the answer is not available in the {{Context:}} and you are not confident about the output,
please say "Information not available in provided context". \n\n
Context: {context}?\n
Question: {question} \n
Answer:
"""

print("[Prompt]")
print(prompt)

print("[Response]")
print(
    generation_model.predict(
        prompt,
        max_output_tokens=256,
        temperature=0.3,
    ).text
)

#Few-shot prompting
prompt = """
Context:
The term "artificial intelligence" was first coined by John McCarthy in 1956. Since then, AI has developed into a vast
field with numerous applications, ranging from self-driving cars to virtual assistants like Siri and Alexa.

Question:
What is artificial intelligence?

Answer:
Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.

---

Context:
The Wright brothers, Orville and Wilbur, were two American aviation pioneers who are credited with inventing and
building the world's first successful airplane and making the first controlled, powered and sustained heavier-than-air human flight,
 on December 17, 1903.

Question:
Who were the Wright brothers?

Answer:
The Wright brothers were American aviation pioneers who invented and built the world's first successful airplane
and made the first controlled, powered and sustained heavier-than-air human flight, on December 17, 1903.

---

Context:
The Mona Lisa is a 16th-century portrait painted by Leonardo da Vinci during the Italian Renaissance. It is one of
the most famous paintings in the world, known for the enigmatic smile of the woman depicted in the painting.

Question:
Who painted the Mona Lisa?

Answer:

"""
print(
    generation_model.predict(
        prompt,
    ).text
)

#Extractive Question-Answering
prompt = """
Background: There is evidence that there have been significant changes in Amazon rainforest vegetation over the last 21,000 years through the Last Glacial Maximum (LGM) and subsequent deglaciation.
Analyses of sediment deposits from Amazon basin paleo lakes and from the Amazon Fan indicate that rainfall in the basin during the LGM was lower than for the present, and this was almost certainly
associated with reduced moist tropical vegetation cover in the basin. There is debate, however, over how extensive this reduction was. Some scientists argue that the rainforest was reduced to small,
isolated refugia separated by open forest and grassland; other scientists argue that the rainforest remained largely intact but extended less far to the north, south, and east than is seen today.
This debate has proved difficult to resolve because the practical limitations of working in the rainforest mean that data sampling is biased away from the center of the Amazon basin, and both
explanations are reasonably well supported by the available data.

Q: What does LGM stands for?
A: Last Glacial Maximum.

Q: What did the analysis from the sediment deposits indicate?
A: Rainfall in the basin during the LGM was lower than for the present.

Q: What are some of scientists arguments?
A: The rainforest was reduced to small, isolated refugia separated by open forest and grassland.

Q: There have been major changes in Amazon rainforest vegetation over the last how many years?
A: 21,000.

Q: What caused changes in the Amazon rainforest vegetation?
A: The Last Glacial Maximum (LGM) and subsequent deglaciation

Q: What has been analyzed to compare Amazon rainfall in the past and present?
A: Sediment deposits.

Q: What has the lower rainfall in the Amazon during the LGM been attributed to?
A:
"""

print(
    generation_model.predict(
        prompt,
    ).text
)

#Evaluation
qa_data = {
    "question": [
        "In a website browser address bar, what does “www” stand for?",
        "Who was the first woman to win a Nobel Prize",
        "What is the name of the Earth’s largest ocean?",
    ],
    "answer_groundtruth": ["World Wide Web", "Marie Curie", "The Pacific Ocean"],
}
qa_data_df = pd.DataFrame(qa_data)
qa_data_df

def get_answer(row):
    prompt = f"""Answer the following question as precise as possible.\n\n
            question: {row}
            answer:
              """
    return generation_model.predict(
        prompt=prompt,
    ).text


qa_data_df["answer_prediction"] = qa_data_df["question"].apply(get_answer)
qa_data_df

#!pip install -q python-Levenshtein --upgrade --user
#!pip install -q fuzzywuzzy --upgrade --user

from fuzzywuzzy import fuzz


def get_fuzzy_match(df):
    return fuzz.partial_ratio(df["answer_groundtruth"], df["answer_prediction"])


qa_data_df["match_score"] = qa_data_df.apply(get_fuzzy_match, axis=1)
qa_data_df

print(
    "the average match score of all predicted answer from PaLM 2 is : ",
    qa_data_df["match_score"].mean(),
    " %",
)