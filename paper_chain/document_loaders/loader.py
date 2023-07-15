import os
from .pdf_downloader import download_pdf
from .pdf_to_txt import convert_pdf_to_txt
from .chunker import chunk_text


def load_documents(url, dest_folder, chunk_size, overlap=0):
    """
    Download a PDF from a URL, convert it to text, and chunk it.

    :param url: The URL of the PDF to download.
    :param dest_folder: The folder where the PDF should be saved.
    :param chunk_size: The size of the chunks.
    :param overlap: The size of the overlap between chunks.
    :return: A tuple containing a list of text chunks and the name of the downloaded file.
    """
    # Download the PDF
    pdf_path = download_pdf(url, dest_folder)
    
    # Get the name of the downloaded file
    filename = os.path.basename(pdf_path)
    
    # Convert the PDF to text
    text = convert_pdf_to_txt(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size, overlap)
    
    return chunks, filename
