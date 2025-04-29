import os
from typing import Dict, Any, List
from google.cloud import vision
import json

from dotenv import load_dotenv

load_dotenv()

# Important fix: Set the absolute path for credentials
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(creds_path)


def analyze_document_layout(file_path: str) -> Dict[str, Any]:
    """
    Analyze document layout using the Vision API Document Text Detection.
    This is an alternative to Document AI that doesn't require entity types.
    """
    client = vision.ImageAnnotatorClient()

    with open(file_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Use DOCUMENT_TEXT_DETECTION for better layout analysis
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            f"Error detecting document text: {response.error.message}")

    return _format_vision_response(response.full_text_annotation)


def _format_vision_response(full_text_annotation) -> Dict[str, Any]:
    """Format the Vision API response to extract layout information."""
    result = {
        "text": full_text_annotation.text,
        "pages": [],
    }

    # Process each page
    for page_idx, page in enumerate(full_text_annotation.pages):
        page_data = {
            "page_number": page_idx + 1,
            "dimensions": {
                "width": page.width,
                "height": page.height,
            },
            "blocks": [],
            "paragraphs": [],
            "lines": [],
            "words": [],
        }

        # Process blocks (similar to Document AI blocks)
        for block_idx, block in enumerate(page.blocks):
            block_text = _get_text_from_block(block)
            block_data = {
                "id": f"block_{block_idx}",
                "text": block_text,
                "confidence": block.confidence,
                "bounding_box": _format_bounding_box(block.bounding_box),
                "paragraphs": [],
            }

            # Process paragraphs within each block
            for para_idx, paragraph in enumerate(block.paragraphs):
                para_text = _get_text_from_paragraph(paragraph)
                para_data = {
                    "id": f"block_{block_idx}_para_{para_idx}",
                    "text": para_text,
                    "confidence": paragraph.confidence,
                    "bounding_box": _format_bounding_box(paragraph.bounding_box),
                    "words": [],
                }

                # Process words within each paragraph
                for word_idx, word in enumerate(paragraph.words):
                    word_text = ''.join(
                        [symbol.text for symbol in word.symbols])
                    word_data = {
                        "id": f"block_{block_idx}_para_{para_idx}_word_{word_idx}",
                        "text": word_text,
                        "confidence": word.confidence,
                        "bounding_box": _format_bounding_box(word.bounding_box),
                        "symbols": [],
                    }

                    # Add symbols (characters) within each word
                    for symbol in word.symbols:
                        symbol_data = {
                            "text": symbol.text,
                            "confidence": symbol.confidence,
                            "bounding_box": _format_bounding_box(symbol.bounding_box),
                        }

                        # Add break information if present
                        if symbol.property and symbol.property.detected_break:
                            break_type = symbol.property.detected_break.type_
                            symbol_data["break_type"] = vision.TextAnnotation.DetectedBreak.BreakType(
                                break_type).name

                        word_data["symbols"].append(symbol_data)

                    para_data["words"].append(word_data)
                    page_data["words"].append(word_data)

                block_data["paragraphs"].append(para_data)
                page_data["paragraphs"].append(para_data)

            page_data["blocks"].append(block_data)

        # Add lines (sequences of words)
        line_idx = 0
        current_line = []
        current_line_box = None

        # Group words into lines based on vertical position
        sorted_words = sorted(page_data["words"], key=lambda w: (
            _get_avg_y(w["bounding_box"]),
            _get_min_x(w["bounding_box"])
        ))

        for word in sorted_words:
            if not current_line:
                current_line.append(word)
                current_line_box = word["bounding_box"].copy()
                continue

            last_word = current_line[-1]
            last_y = _get_avg_y(last_word["bounding_box"])
            curr_y = _get_avg_y(word["bounding_box"])

            # If the word is roughly on the same line (within 10 pixels)
            if abs(curr_y - last_y) < 10:
                current_line.append(word)
                # Expand bounding box
                current_line_box = _merge_bounding_boxes(
                    current_line_box, word["bounding_box"])
            else:
                # Create a new line
                line_data = {
                    "id": f"line_{line_idx}",
                    "text": " ".join([w["text"] for w in current_line]),
                    "bounding_box": current_line_box,
                    "words": [w["id"] for w in current_line]
                }
                page_data["lines"].append(line_data)
                line_idx += 1

                # Start a new line
                current_line = [word]
                current_line_box = word["bounding_box"].copy()

        # Don't forget the last line
        if current_line:
            line_data = {
                "id": f"line_{line_idx}",
                "text": " ".join([w["text"] for w in current_line]),
                "bounding_box": current_line_box,
                "words": [w["id"] for w in current_line]
            }
            page_data["lines"].append(line_data)

        result["pages"].append(page_data)

    return result


def _get_text_from_block(block) -> str:
    """Extracting text from a block by concatenating paragraph texts."""
    return " ".join([_get_text_from_paragraph(paragraph) for paragraph in block.paragraphs])


def _get_text_from_paragraph(paragraph) -> str:
    """Extracting text from a paragraph by concatenating word texts with appropriate spacing."""
    text = ""
    for word in paragraph.words:
        word_text = ""
        for symbol in word.symbols:
            word_text += symbol.text
            if symbol.property and symbol.property.detected_break:
                break_type = symbol.property.detected_break.type_
                if break_type == vision.TextAnnotation.DetectedBreak.BreakType.SPACE:
                    word_text += " "
                elif break_type == vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK:
                    word_text += "\n"
                elif break_type == vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE:
                    word_text += " \n"

        text += word_text

    return text


def _format_bounding_box(bounding_box) -> List[Dict[str, int]]:
    """Convert Vision API bounding box to standardized format."""
    return [{"x": vertex.x, "y": vertex.y} for vertex in bounding_box.vertices]


def _get_avg_y(bounding_box: List[Dict[str, int]]) -> float:
    """Get the average Y coordinate of a bounding box."""
    return sum(point["y"] for point in bounding_box) / len(bounding_box)


def _get_min_x(bounding_box: List[Dict[str, int]]) -> int:
    """Get the minimum X coordinate of a bounding box."""
    return min(point["x"] for point in bounding_box)


def _merge_bounding_boxes(box1: List[Dict[str, int]], box2: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """Merge two bounding boxes into one that encompasses both."""
    min_x = min(min(p["x"] for p in box1), min(p["x"] for p in box2))
    min_y = min(min(p["y"] for p in box1), min(p["y"] for p in box2))
    max_x = max(max(p["x"] for p in box1), max(p["x"] for p in box2))
    max_y = max(max(p["y"] for p in box1), max(p["y"] for p in box2))

    return [
        {"x": min_x, "y": min_y},  # top-left
        {"x": max_x, "y": min_y},  # top-right
        {"x": max_x, "y": max_y},  # bottom-right
        {"x": min_x, "y": max_y}   # bottom-left
    ]


def main():
    """Test the Vision API document layout analysis functionality."""
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python vision_layout_reader.py [file_path]")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        result = analyze_document_layout(file_path)
        print(f"Successfully analyzed document: {file_path}")
        print(f"Total pages: {len(result['pages'])}")
        print(f"Extracted text (first 100 chars): {result['text'][:100]}...")

        if result["pages"]:
            first_page = result["pages"][0]
            print(f"Blocks on first page: {len(first_page['blocks'])}")
            print(f"Paragraphs on first page: {len(first_page['paragraphs'])}")
            print(f"Lines on first page: {len(first_page['lines'])}")
            print(f"Words on first page: {len(first_page['words'])}")

            # Write results to JSON file for inspection
            output_file = f"{os.path.splitext(file_path)[0]}_layout.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Full analysis saved to: {output_file}")

    except Exception as e:
        print(f"Error analyzing document: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
