#Install Vertex AI SDK
#!pip install google-cloud-aiplatform --upgrade --user

#Import libraries
import pandas as pd
from vertexai.language_models import TextGenerationModel

#Import models
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Zero-shot prompting
prompt = """
Classify the following:\n
text: "I saw a furry animal in the park today with a long tail and big eyes."
label: dogs, cats
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Few-shot prompting
prompt = """
What is the topic for a given news headline? \n
- business \n
- entertainment \n
- health \n
- sports \n
- technology \n\n

Text: Pixel 7 Pro Expert Hands On Review. \n
The answer is: technology \n

Text: Quit smoking? \n
The answer is: health \n

Text: Birdies or bogeys? Top 5 tips to hit under par \n
The answer is: sports \n

Text: Relief from local minimum-wage hike looking more remote \n
The answer is: business \n

Text: You won't guess who just arrived in Bari, Italy for the movie premiere. \n
The answer is:
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Topic classification
prompt = """
Classify a piece of text into one of several predefined topics, such as sports, politics, or entertainment. \n
text: President Biden will be visiting India in the month of March to discuss a few opportunites. \n
class:
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Spam detection
prompt = """
Given an email, classify it as spam or not spam. \n
email: hi user, \n
      you have been selected as a winner of the lotery and can win upto 1 million dollar. \n
      kindly share your bank details and we can proceed from there. \n\n

      from, \n
      US Official Lottry Depatmint
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Intent recognition
prompt = """
Given a user's input, classify their intent, such as "finding information", "making a reservation", or "placing an order". \n
user input: Hi, can you please book a table for two at Juan for May 1?
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Language identification
prompt = """
Given a piece of text, classify the language it is written in. \n
text: Selam nasıl gidiyor?
language:
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Toxicity detection
prompt = """
Given a piece of text, classify it as toxic or non-toxic. \n
text: i love sunny days
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Emotion detection
prompt = """
Given a piece of text, classify the emotion it conveys, such as happiness, or anger. \n
text: I'm still so delighted from yesterday's news
"""

print(
    generation_model.predict(
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.1,
    ).text
)

#Evaluation
review_data = {
    "review": [
        "i love this product. it does have everything i am looking for!",
        "all i can say is that you will be happy after buying this product",
        "its way too expensive and not worth the price",
        "i am feeling okay. its neither good nor too bad.",
    ],
    "sentiment_groundtruth": ["positive", "positive", "negative", "neutral"],
}

review_data_df = pd.DataFrame(review_data)
review_data_df

def get_sentiment(row):
    prompt = f"""Classify the sentiment of the following review as "positive", "neutral" and "negative". \n\n
                review: {row} \n
                sentiment:
              """
    response = generation_model.predict(prompt=prompt).text
    return response


review_data_df["sentiment_prediction"] = review_data_df["review"].apply(get_sentiment)
review_data_df

from sklearn.metrics import classification_report

print(
    classification_report(
        review_data_df["sentiment_groundtruth"], review_data_df["sentiment_prediction"]
    )
)