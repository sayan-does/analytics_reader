import os
from google.cloud import documentai_v1 as documentai
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# Important fix: Set the absolute path for credentials
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(creds_path)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
location = os.getenv("GCP_LOCATION")  # Default to 'us' if not set


def analyze_layout(file_path: str) -> Dict[str, Any]:
    client = documentai.DocumentProcessorServiceClient()

    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    with open(file_path, "rb") as file:
        content = file.read()

    raw_document = documentai.RawDocument(
        content=content, mime_type=_get_mime_type(file_path))

    # entity_types = [
    #     documentai.EntityType(type_="Name"),
    #     documentai.EntityType(type_="Date"),
    # ]

    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )

    response = client.process_document(request=request)

    return _format_layout_response(response.document)


def _get_mime_type(file_path: str) -> str:
    """Determine mime type based on file extension."""
    extension = file_path.lower().split(".")[-1]
    mime_types = {
        "pdf": "application/pdf",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "gif": "image/gif",
    }
    
    return mime_types.get(extension, "application/pdf")


def _format_layout_response(document) -> Dict[str, Any]:
    """Format the Document AI response to extract layout information."""
    result = {
        "text": document.text,
        "pages": [],
        "entities": [],
    }
    
    for page in document.pages:
        page_data = {
            "page_number": page.page_number,
            "dimensions": {
                "width": page.dimension.width,
                "height": page.dimension.height,
            },
            "blocks": [],
            "paragraphs": [],
            "lines": [],
            "tokens": [],
        }
        
        for block in page.blocks:
            block_data = {
                "layout": _extract_layout_info(block.layout),
                "text": _get_text_from_layout(document.text, block.layout)
            }
            page_data["blocks"].append(block_data)
            
        for paragraph in page.paragraphs:
            paragraph_data = {
                "layout": _extract_layout_info(paragraph.layout),
                "text": _get_text_from_layout(document.text, paragraph.layout)
            }
            page_data["paragraphs"].append(paragraph_data)
            
        for line in page.lines:
            line_data = {
                "layout": _extract_layout_info(line.layout),
                "text": _get_text_from_layout(document.text, line.layout)
            }
            page_data["lines"].append(line_data)
            
        for token in page.tokens:
            token_data = {
                "layout": _extract_layout_info(token.layout),
                "text": _get_text_from_layout(document.text, token.layout)
            }
            page_data["tokens"].append(token_data)
        
        result["pages"].append(page_data)
    
    for entity in document.entities:
        result["entities"].append({
            "type": entity.type_,
            "text": _get_text_from_layout(document.text, entity.page_anchor),
            "confidence": entity.confidence,
        })
    
    return result


def _extract_layout_info(layout) -> Dict[str, Any]:
    """Extract layout information from Document AI layout object."""
    if not layout or not layout.bounding_poly:
        return {}
    
    vertices = []
    for vertex in layout.bounding_poly.vertices:
        vertices.append({"x": vertex.x, "y": vertex.y})
    
    return {
        "confidence": layout.confidence,
        "bounding_box": vertices
    }


def _get_text_from_layout(text: str, layout) -> str:
    """Extract text content based on text anchor information."""
    if not layout or not hasattr(layout, "text_anchor"):
        return ""
    
    segments = layout.text_anchor.text_segments
    if not segments:
        return ""
    
    text_parts = []
    for segment in segments:
        start_index = segment.start_index
        end_index = segment.end_index
        text_parts.append(text[start_index:end_index])
    
    return "".join(text_parts)


def main():
    """Test the Document AI layout analysis functionality."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python layout_reader.py [file_path]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        result = analyze_layout(file_path)
        print(f"Successfully analyzed document: {file_path}")
        print(f"Total pages: {len(result['pages'])}")
        print(f"Extracted text (first 100 chars): {result['text'][:100]}...")
        
        if result["pages"]:
            first_page = result["pages"][0]
            print(f"First page dimensions: {first_page['dimensions']['width']} x {first_page['dimensions']['height']}")
            print(f"Blocks on first page: {len(first_page['blocks'])}")
            print(f"Paragraphs on first page: {len(first_page['paragraphs'])}")
    except Exception as e:
        print(f"Error analyzing document: {str(e)}")


if __name__ == "__main__":
    main()