import os, sys, time
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geminipdf.parse import parse_pdf

# Load from environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key = os.getenv("GOOGLE_API_KEY")

def parse_arguments():
    """
    Parse command line arguments
    :return: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="PDF to Text Conversion using GPT")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--model', type=str, 
                       default='gemini-2.0-flash-exp',
                       help='default gemini model to use')
    return parser.parse_args()

def extract_outdir_from_path(pdf_path: str) -> str:
    try:
        output_dir = os.path.splitext(pdf_path)[0]  # Remove file extension
        return output_dir
    except:
        return './output'

if __name__ == "__main__":
    args = parse_arguments()
    output_dir = extract_outdir_from_path(args.pdf_path)
    start_time = time.time()
    print('waiting...\n')
    content, image_paths = parse_pdf(
        pdf_path=args.pdf_path,
        output_dir=output_dir,
        model=args.model,
        api_key=api_key
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')