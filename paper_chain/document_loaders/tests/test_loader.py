from document_loaders.loader import load_documents

def test_load_documents():
    # Specify the URL of the PDF to download
    url = "https://www.roboticsproceedings.org/rss19/p001.pdf"

    # Specify the destination folder for the PDF
    dest_folder = "paper_chain/data/pdfs"

    # Specify the chunk size
    chunk_size = 5000

    # Call the load_documents function
    chunks = load_documents(url, dest_folder, chunk_size)

    # Check the output
    assert isinstance(chunks, list), "Output should be a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "Each chunk should be a string"
    print(chunks)
    print("Test passed!")

# Run the test
test_load_documents()
