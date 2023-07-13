from document_loaders.chunker import chunk_text

# Specify the text to chunk
text = "This is a long text that will be split into chunks."

# Call the function to be tested
chunks = chunk_text(text, chunk_size=10)

# Check the result
for chunk in chunks:
    print(chunk)
