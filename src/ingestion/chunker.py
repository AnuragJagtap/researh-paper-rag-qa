import re

def split_into_sentences(text):
    """
    Splits text into sentences using regex
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Creates overlapping chunks while preserving sentence structure
    """

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding sentence exceeds chunk size → finalize chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())

            # Create overlap
            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks