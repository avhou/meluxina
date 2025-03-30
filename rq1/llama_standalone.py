import transformers
import torch
from models import *
from datetime import datetime

model_id = "meta-llama/Llama-3.3-70B-Instruct"

print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

messages = [
    {"role": "system", "content": f"""
    You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.
    """},
    {"role": "user", "content": """
    Premium for mayors: 2,000 to 3,500 euros per collected asylum seeker that I was also thinking of yes. Second is also true of course, but it is a premium, not an expense allowance. Have they already tried https://m.Nieuwsblad.be/cnt/DMF20160218_02135844 Sorry that it is about Ukrainians, was just the first article I found we can't give 3,500 euros to those asylum seekers to get it back? Does that work with retroactive effect? Just Asking for a Friend - Antwerp. I mean, we don't have the mean to handle this coply, and if there are no clean reception/prospects you'll just have (novelhelmingly) Men turn to crime. > Equipped with sufficient care = local this here is completely wrong iMop. There is a reason why Fedasil exists. A well -coordinated and organized approach for a macro given as the asylum crisis cannot be expected from Middleofnowheregem and even from the big cities is unreasonable. It is simply their responsibility/powers to move on as they have been doing for decades. As you can read in the article, by the way, because the cities and municipalities do not know what to do with the current situation. Belgium is more than an administrative processing organ, normally a "policy" is also expected .. We could always try a long -term vision instead of playing winds with the political wind, but that is too ambitious .. * delegate * it is called that ... /s> the premiums do not seem to convince the municipalities. "We are not concerned with the money," says Nathalie Debast, spokeswoman for the Association of Flemish Cities and Municipalities (VVSG). "The rack is just out, after years of jojoble policy. First the municipalities had to create extra places, then reduce again and now more. Moreover, there are hardly any places. The municipal authorities should rent on the private market, but then they compete with their own residents, who are already struggling to find a rental home." Pretty the most important thing about the article. Just add more money to the local authorities until the problem leaves or resolved is available. "Again such a clickbait title, Tis a premium for the municipality, not for the mayor personally. Also think that € 3000 peanuts is to organize and support the reception of a person. Certainly with the current prices of gas and electricity. with ten in a room of four. "
    """},
]

outputs = pipeline(
    messages,
    max_new_tokens=1000,
)
print(outputs[0]["generated_text"][-1])
