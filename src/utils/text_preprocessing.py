import re

def clean_query(query):
    # Lowercase
    query = query.lower()

    # Remove extra spaces
    query = re.sub(r"\s+", " ", query)

    return query.strip()