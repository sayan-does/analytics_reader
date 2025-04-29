import os
import base64
from typing import Dict, Any, List, Optional, Union
from google.cloud import vision

import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")

language_hints = os.getenv("OCR_LANGUAGE_HINTS", "en").split(",")

detect_orientation = os.getenv("OCR_DETECT_ORIENTATION", "true").lower() == "true"
def extract_text(
    image_path: str,
    language_hints: List[str] = None,
    detect_orientation: bool = False
) -> Dict[str, Any]:
    
    client = vision.ImageAnnotatorClient()
    
    if image_path.startswith('data:image'):
        image_content = base64.b64decode(image_path.split(',')[1])
        image = vision.Image(content=image_content)
    elif os.path.isfile(image_path):
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
        image = vision.Image(content=image_content)
    else:
        raise ValueError("Invalid image path or base64 string")
    
    features = [vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]
    
    image_context = {}
    if language_hints:
        image_context["language_hints"] = language_hints
    if detect_orientation:
        image_context["text_detection_params"] = {"enable_text_detection_confidence_score": True}
    
    request = {
        "image": image,
        "features": features
    }
    
    if image_context:
        request["image_context"] = vision.ImageContext(**image_context)
    
    response = client.annotate_image(request=request)
    
    if response.error.message:
        raise Exception(f"Error detecting text: {response.error.message}")
    
    return _format_text_annotations(response.text_annotations, response.full_text_annotation)


def _format_text_annotations(text_annotations, full_text_annotation) -> Dict[str, Any]:
    result = {
        "text": "",
        "locale": "",
        "blocks": [],
    }
    
    if text_annotations:
        result["text"] = text_annotations[0].description
        result["locale"] = text_annotations[0].locale
    
    if full_text_annotation:
        for page in full_text_annotation.pages:
            for block in page.blocks:
                block_data = {
                    "text": _get_text_from_block(block),
                    "confidence": block.confidence,
                    "bounding_box": _format_bounding_poly(block.bounding_box),
                    "paragraphs": []
                }
                
                for paragraph in block.paragraphs:
                    paragraph_data = {
                        "text": _get_text_from_paragraph(paragraph),
                        "confidence": paragraph.confidence,
                        "bounding_box": _format_bounding_poly(paragraph.bounding_box),
                        "words": []
                    }
                    
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        word_data = {
                            "text": word_text,
                            "confidence": word.confidence,
                            "bounding_box": _format_bounding_poly(word.bounding_box),
                            "symbols": []
                        }
                        
                        for symbol in word.symbols:
                            symbol_data = {
                                "text": symbol.text,
                                "confidence": symbol.confidence,
                                "bounding_box": _format_bounding_poly(symbol.bounding_box),
                            }
                            
                            if symbol.property and symbol.property.detected_break:
                                break_type = symbol.property.detected_break.type_
                                symbol_data["break_type"] = vision.TextAnnotation.DetectedBreak.BreakType(break_type).name
                            
                            word_data["symbols"].append(symbol_data)
                        
                        paragraph_data["words"].append(word_data)
                    
                    block_data["paragraphs"].append(paragraph_data)
                
                result["blocks"].append(block_data)
    
    return result


def _format_bounding_poly(bounding_poly) -> List[Dict[str, float]]:
    """Converting bounding polygon to a list of vertex dictionaries."""
    return [{"x": vertex.x, "y": vertex.y} for vertex in bounding_poly.vertices]


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


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: text_reader.py [image_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        result = extract_text(image_path)
        print(f"Successfully extracted text from: {image_path}")
        print(f"Detected locale: {result['locale']}")
        print(f"Extracted text:\n{result['text'][:300]}...")
        print(f"Number of text blocks: {len(result['blocks'])}")
        
        if result['blocks']:
            first_block = result['blocks'][0]
            print(f"First block text: {first_block['text'][:100]}...")
            print(f"First block confidence: {first_block['confidence']}")
    except Exception as e:
        print(f"Error extracting text: {str(e)}")


if __name__ == "__main__":
    main()