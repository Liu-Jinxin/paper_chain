def chunk_text(text, chunk_size):
    """
    Split text into chunks of a given size.
    
    :param text: The text to chunk.
    :param chunk_size: The maximum size of each chunk.
    :return: A list of text chunks.
    """
    words = text.split()
    chunks = []
    
    # While there are still words left
    while words:
        # Initialize the chunk and its size
        chunk = []
        size = 0
        
        # While the size of the chunk is less than the chunk size and there are still words left
        while words and size + len(words[0]) <= chunk_size:
            word = words.pop(0)
            chunk.append(word)
            size += len(word)
        
        # Join the chunk and add it to the list of chunks
        chunks.append(' '.join(chunk))
    
    return chunks
