import requests
import os
from urllib.parse import urlsplit

def download_pdf(url, dest_folder):
    """
    Download a PDF from a URL and save it to a destination folder.
    
    :param url: The URL of the PDF to download.
    :param dest_folder: The folder where the PDF should be saved.
    """
    # Make sure that the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Make sure that the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF from {url}. Status code: {response.status_code}")
    
    # Get the filename from the URL
    filename = os.path.basename(urlsplit(url).path)
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # Write the content of the response to a file in the destination folder
    file_path = os.path.join(dest_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    return file_path
