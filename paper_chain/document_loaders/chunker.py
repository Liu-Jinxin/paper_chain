def chunk_text(text, chunk_size, overlap=0, max_chunks=None):
    """
    Split text into chunks of a given size.

    :param text: The text to chunk.
    :param chunk_size: The maximum size of each chunk.
    :param overlap: The size of the overlap between chunks.
    :param max_chunks: The maximum number of chunks to return. If not None, the function will return at most 
                       'max_chunks' chunks, and discard the remaining text.
    :return: A list of text chunks.
    """
    words = text.split()
    chunks = []
    
    # While there are still words left and the maximum number of chunks has not been reached
    while words and (max_chunks is None or len(chunks) < max_chunks):
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
        
        # If there is overlap, put back the last overlap number of words back into words list
        if overlap > 0 and words:
            words = chunk[-overlap:] + words
    
    return chunks
