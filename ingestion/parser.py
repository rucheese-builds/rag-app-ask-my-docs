import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument
import pdfplumber

# Load environment variables
load_dotenv()

def format_table_as_markdown(table):
    """Convert a table represented as a list of lists into a Markdown table."""
    if not table or not any(table):
        return ""
    # Filter out empty rows/cols
    table = [[(cell or "").strip() for cell in row] for row in table]
    table = [row for row in table if any(row)]
    if not table:
        return ""

    md_lines = []
    # Header row
    headers = [x.replace("\n", " ") for x in table[0]]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Data rows
    for row in table[1:]:
        # Ensure row matches header length
        cells = [x.replace("\n", " ") for x in row]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        md_lines.append("| " + " | ".join(cells) + " |")
        
    return "\n\n" + "\n".join(md_lines) + "\n\n"

def parse_with_pdfplumber(pdf_path: Path) -> list[LangchainDocument]:
    """Parse PDF locally using pdfplumber, preserving text and extracting tables to Markdown."""
    print(f"[Parser] Using local pdfplumber for {pdf_path.name}")
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text(layout=False) or ""
            
            # Extract tables on this page
            tables = page.extract_tables()
            table_markdowns = []
            for table in tables:
                tb_md = format_table_as_markdown(table)
                if tb_md:
                    table_markdowns.append(tb_md)
            
            # Combine text and table markdown
            combined_content = text
            if table_markdowns:
                combined_content += "\n" + "\n".join(table_markdowns)
                
            doc = LangchainDocument(
                page_content=combined_content,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "parser": "pdfplumber"
                }
            )
            documents.append(doc)
            
    return documents

def parse_with_llamaparse(pdf_path: Path) -> list[LangchainDocument]:
    """Parse PDF using LlamaParse API."""
    print(f"[Parser] Using LlamaParse API for {pdf_path.name}")
    from llama_parse import LlamaParse
    
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en"
    )
    
    # LlamaParse load_data parses the document
    llama_docs = parser.load_data(str(pdf_path))
    
    documents = []
    for i, l_doc in enumerate(llama_docs):
        doc = LangchainDocument(
            page_content=l_doc.text,
            metadata={
                "source": pdf_path.name,
                "page": l_doc.metadata.get("page_number", i + 1),
                "parser": "llamaparse"
            }
        )
        documents.append(doc)
        
    return documents

def prune_references(documents: list[LangchainDocument]) -> list[LangchainDocument]:
    """Prune References, Bibliography, and subsequent sections from academic papers."""
    if not documents:
        return documents
        
    import re
    # Scan from page index 2 (page 3) onwards
    for i in range(2, len(documents)):
        content = documents[i].page_content
        # Match standalone line for References, Bibliography, etc.
        pattern = r'(?i)\n(?:#+\s+)?(?:References|Bibliography|Literature\s+Cited)(?:\s+and\s+Notes)?\s*(?::)?\s*(?:\n|$)'
        
        # Prepend \n to match if it starts at the first character of the page
        match = re.search(pattern, '\n' + content)
        if match:
            match_start = match.start()
            if match_start > 0:
                match_start -= 1
            
            # Truncate content of this page at references heading start
            pruned_content = content[:match_start].strip()
            print(f"[Parser] Pruning references section on page {documents[i].metadata.get('page')} of {documents[i].metadata.get('source')}")
            
            if pruned_content:
                documents[i].page_content = pruned_content
                return documents[:i+1]
            else:
                return documents[:i]
                
    return documents

def parse_pdf(pdf_path: Path) -> list[LangchainDocument]:
    """Parse a PDF document based on available environment configuration."""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    
    if api_key and api_key.strip():
        try:
            docs = parse_with_llamaparse(pdf_path)
        except Exception as e:
            print(f"[Parser] LlamaParse failed for {pdf_path.name}: {e}. Falling back to pdfplumber...")
            docs = parse_with_pdfplumber(pdf_path)
    else:
        docs = parse_with_pdfplumber(pdf_path)
        
    # Apply references pruning strictly to research papers
    name = pdf_path.name.lower()
    is_research = not ("earning" in name or "transcript" in name or "nvidiaan" in name)
    if is_research:
        docs = prune_references(docs)
        
    return docs

if __name__ == "__main__":
    # Test parser
    test_pdf = Path("data/research/ReACT.pdf")
    if test_pdf.exists():
        docs = parse_pdf(test_pdf)
        print(f"Parsed {len(docs)} pages.")
        if docs:
            print("--- First 500 chars of page 1 ---")
            print(docs[0].page_content[:500])
    else:
        print(f"Test file {test_pdf} not found.")
