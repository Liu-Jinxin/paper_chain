from .pdf_downloader import download_pdf
from .pdf_to_txt import convert_pdf_to_txt
from .chunker import chunk_text

def load_documents(url, dest_folder, chunk_size):
    """
    Download a PDF from a URL, convert it to text, and chunk it.
    
    :param url: The URL of the PDF to download.
    :param dest_folder: The folder where the PDF should be saved.
    :param chunk_size: The size of the chunks.
    :return: A list of text chunks.
    """
    # Download the PDF
    pdf_path = download_pdf(url, dest_folder)
    
    # Convert the PDF to text
    text = convert_pdf_to_txt(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size)
    
    return chunks
