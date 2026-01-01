from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# loading the api key from my .env file
load_dotenv()

# performing sanity check to see if the key exists
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: API Key is not found. Check that .env file")
else:
    print(f"API Key found: {api_key[:5]}... (hiding the rest from prying eyes)")

# need to initialize the ol' GPT.. aka The Brain
# using temperature=0.7 for a little creativity
# temperature controls the randomness and creativity of the response
# values must be between 0.0 and 2.0.  Default value is 1.0
# It adjusts the probability distribution over the possible next tokens (words or sub-words) the model can select.
# Lower values make the output more deterministic and focused, as the model consistently picks the most probable tokens. 
# A temperature of 0.0 will result in highly predictable responses, useful for factual extraction or structured tasks
# Higher values increase the probability of selecting less likely tokens, leading to more diverse, creative, and sometimes nonsensical or "hallucinated" outputs
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# let's ask it a question
print("asking the AI overlord a question...")
response = llm.invoke("Tell me a short joke about Python programming")

# let us see what is given
print("-" * 20)
print(response.content)
print("-" * 20)