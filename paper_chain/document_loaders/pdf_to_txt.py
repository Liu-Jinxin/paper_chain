from pdfminer.high_level import extract_text

def convert_pdf_to_txt(file_path):
    """
    Convert a PDF to text.
    
    :param file_path: The path to the PDF file.
    :return: The text extracted from the PDF.
    """
    # Use PDFMiner's extract_text function to extract text from the PDF
    text = extract_text(file_path)
    
    return text
