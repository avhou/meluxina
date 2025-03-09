import transformers
from datetime import datetime
import sys

start = datetime.now()
current_time = start.strftime("%H:%M:%S")
print(f"Starting at {current_time}", flush=True)

print(f"found HUGGINGFACE_HUB_CACHE : {sys.env['HUGGINGFACE_HUB_CACHE']}", flush=True)
print(f"found HF_HOME : {sys.env['HF_HOME']}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {sys.env['HUGGINGFACEHUB_API_TOKEN']}", flush=True)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model="mistralai/Mistral-Small-24B-Instruct-2501",
#     model_kwargs={"torch_dtype": "auto"},
#     device_map="auto",
# )


pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

# messages = [
#     {"role": "system", "content": "You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.  You are designed take in any input text, extract the relevant information, and output a knowledge graph in turtle (TTL) format.  Do not provide any explanation or justification.  Only output TTL knowledge graphs."},
#     {"role": "user", "content": """
#     Het Vlaams Belang is er niet over te spreken dat de paars-groene regering zonder enige vorm van discussie miljarden euro’s plant uit te geven aan leeflonen voor gevluchte Oekraïners.
#     “De geraamde kosten bedragen 665 miljoen euro tot einde juli. Dit is gewoonweg onbetaalbaar”, zegt volksvertegenwoordiger Wouter Vermeersch.
#     “Mensen die oorlogen ontvluchten moeten geholpen worden. Maar daar is geen leefloon voor nodig.
#     Dit kan door, naar Nederlands voorbeeld, te voorzien in bed, bad en brood.”
#     In antwoord op vragen van Vermeersch stelde staatssecretaris voor Begroting Eva De Bleeker uit te gaan van een totale instroom van 259.000 vluchtelingen tot de zomer.
#     Hiervan zou 20 procent nood hebben aan opvang.
#     “De geraamde kostprijs van de opvang bedraagt tot eind juli 49 miljoen euro”, aldus Vermeersch.
#     “In de komende maanden wordt de instroom gradueel opgebouwd.
#     De Bleeker raamt de kosten voor leeflonen op 665 miljoen euro tot einde juli.”
#     “259.000 Oekraïners kosten 259 miljoen euro per maand of 3,1 miljard euro per jaar aan leeflonen alleen, dat is onbetaalbaar”.
#     "Als we 200.000 Oekraïense vluchtelingen gemiddeld 1.000 euro per maand leefloon geven, dan kost dit 200 miljoen euro per maand en dus 2,4 miljard euro per jaar.
#     Dat kunnen de overheidsfinanciën gewoon niet aan.
#     Het zal op termijn betaald worden door alle werkende mensen in ons land”, vervolgt Vermeersch.
#     “En dat zijn niet mijn woorden, maar deze van een partijgenoot van De Bleeker, Kamerlid Tim Vandenput (Open Vld).
#     Helaas heeft De Bleeker zelfs hier geen boodschap aan, ze gaat er gewoon niet op in.”.
#     “Als een aantal van die cijfers hier worden bevestigd, zoals het cijfer van de verwachte 259.000 vluchtelingen, dan is het door de partijgenoten van de staatssecretaris berekende cijfer van 2,4 miljard aan leeflonen per jaar zeer realistisch”, besluit Vermeersch.
#     “Gevluchte Oekraïners moeten een veilig onderkomen krijgen, evenals de nodige medische en sociale hulp.
#     Leeflonen zijn onnodig.”
# """},
# ]
#
messages = [
    {"role": "system", "content": "You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.  You are designed take in any input text, extract the relevant information, and output a knowledge graph in turtle (TTL) format.  Do not provide any explanation or justification.  Only output TTL knowledge graphs."},
    {"role": "user", "content": """
Vlaams Belang strongly disagrees with the purple-Green government planning billions of euros without any discussion on living wages for refugees Ukrainians.
About 665 million euros until the end of July.
This is simply priceless debt, says representative Wouter vermeersch.
Paid-up benefit is not required for this purpose.
This can be done by providing, according to Dutch example, in bed, bath and bread.
Since Vermeersch's questions, the Secretary of State for Budget Eva De Bleeker suggested that he should assume a total influx of 259 000 refugees until summer.
Of these 20 percent would need shelter.
About 49 million euros of cash benefits are calculated until the end of July, says Vermeersch.
In the coming months, the influx is gradually being built up.
De Bleeker estimates the cost of living wages at 665 million euros until the end of July.
259.000 Ukrainians cost 259 million euro per month or 3.1 billion euros per year in living wages alone, which is unaffordable "If we give 200,000 Ukrainian refugees an average of 1,000 euros a month, this costs 200 million euro/month and therefore 2.4 billion euro per year.
That's just not possible for public finances. It will be paid in due course by all working people in our country.
Unfortunately, De Bleeker does not even take a message here, she simply does not go into it.
• If some of these figures are confirmed here, such as the figure of the expected 259 000 refugees, then the figure calculated by the party members of the State Secretary of State is 2.4 billion in living wages per year very realistic: Vermeersch decides.
♪Flighted Ukrainians must get safe accommodation, as well as the necessary medical and social assistance.
Living wages are unnecessary.
"""},
]

# messages = [
#     {"role": "system", "content": "You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.  You are designed take in any input text, extract the relevant information, and output a knowledge graph in turtle (TTL) format.  Do not provide any explanation or justification.  Only output TTL knowledge graphs."},
#     {"role": "user", "content": """
# Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
# She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
# Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
# She was, in 1906, the first woman to become a professor at the University of Paris.
# """},
# ]

outputs = pipeline(messages, max_new_tokens=512)
print(outputs[0]["generated_text"][-1], flush=True)


stop = datetime.now()
current_time = stop.strftime("%H:%M:%S")
print(f"Stopping at {current_time}", flush=True)


# from accelerate import Accelerator
# accelerator = Accelerator()
#
# # This will parallelize the execution across devices (GPUs/TPUs) and CPU if necessary
# chunks = [{"role": "user", "content": chunk} for chunk in text_chunks]
# outputs = accelerator.gather(pipeline(chunks))
