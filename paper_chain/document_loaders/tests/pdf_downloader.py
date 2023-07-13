import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from document_loaders.pdf_downloader import download_pdf

# Specify the URL of the PDF file to download
url = 'https://www.roboticsproceedings.org/rss19/p001.pdf'
current_dir = os.path.dirname(os.path.abspath(__file__))
dest_folder = os.path.join(current_dir, '../../data/pdfs')

# Call the function to be tested
pdf_path = download_pdf(url, dest_folder)

# Check the result
print(f'Downloaded file is stored in: {pdf_path}')
