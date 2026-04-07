import spacy

nlp = spacy.load("en_core_web_sm")

def extract_triplets(text):
    doc = nlp(text)

    triplets = []

    for token in doc:
        if token.dep_ == "ROOT":
            subject = None
            obj = None

            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child.text
                if child.dep_ in ["dobj", "attr", "pobj"]:
                    obj = child.text

            if subject and obj:
                triplets.append((subject, token.text, obj))

    return triplets