import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from document_loaders.pdf_to_txt import convert_pdf_to_txt

current_dir = os.path.dirname(os.path.abspath(__file__))
dest_folder = os.path.join(current_dir, '../../data/pdfs')
# Specify the path of the PDF file to convert
pdf_path = dest_folder + '/' + 'p001.pdf'

# Call the function to be tested
text = convert_pdf_to_txt(pdf_path)

# Check the result
print(text)
