import os
import json
import base64
from google.cloud import vision
from google.cloud import documentai_v1 as documentai
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class TextBlock:
    text: str
    confidence: float
    bounding_box: List[Tuple[int, int]]
    block_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentStructure:
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[TextBlock] = field(default_factory=list)
    form_fields: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string with custom serialization for TextBlocks"""
        return json.dumps(asdict(self), indent=2)

class GoogleCloudVisionProcessor:
    """Process documents using Google Cloud Vision API and Document AI.
    
    This class provides methods to extract text, structure, tables, and form fields
    from document images using Google Cloud services. It can utilize either the 
    Vision API for basic document understanding or Document AI for more advanced
    document processing capabilities if a processor ID is provided.
    """
    
    def __init__(self, project_id=None, location=None, processor_id=None, language_hints=None):
        """Initialize the processor with Google Cloud credentials and settings."""
        # Read from environment variables if not provided
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location = location or os.environ.get('GCP_LOCATION', 'us')
        self.processor_id = processor_id or os.environ.get('DOCUMENT_AI_PROCESSOR_ID')
        self.language_hints = language_hints or os.environ.get('OCR_LANGUAGE_HINTS', 'en').split(',')
        self.detect_orientation = os.environ.get('OCR_DETECT_ORIENTATION', 'False').lower() == 'true'
        
        # Initialize Vision API client
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize Document AI client if processor_id is available
        self.docai_client = None
        if self.processor_id:
            self.docai_client = documentai.DocumentProcessorServiceClient()
            self.processor_name = self.docai_client.processor_path(
                self.project_id, self.location, self.processor_id
            )
    
    def process_document(self, file_path):
        """Process a document image and extract structured information."""
        # If Document AI processor is configured, use it
        if self.docai_client and self.processor_id:
            return self._process_with_document_ai(file_path)
        
        # Otherwise, use Vision API
        return self._process_with_vision_api(file_path)
    
    def _process_with_vision_api(self, file_path):
        """Process document using Vision API for OCR and structure detection."""
        document = DocumentStructure()
        
        # Read the file
        with open(file_path, "rb") as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Configure image context with language hints and orientation
        image_context = vision.ImageContext(
            language_hints=self.language_hints,
            text_detection_params=vision.TextDetectionParams(
                enable_text_detection_confidence_score=True
            )
        )
        
        # Detect text
        text_detection_response = self.vision_client.text_detection(
            image=image, 
            image_context=image_context
        )
        if text_detection_response.error.message:
            raise Exception(f"Error detecting text: {text_detection_response.error.message}")
        
        # Process full text annotation
        document = self._process_text_annotation(document, text_detection_response.full_text_annotation)
        
        # Detect document features
        document_response = self.vision_client.document_text_detection(
            image=image,
            image_context=image_context
        )
        if document_response.error.message:
            raise Exception(f"Error detecting document: {document_response.error.message}")
        
        # Extract form fields from document
        self._extract_form_fields(document, document_response.full_text_annotation)
        
        # Use document text detection for layout analysis
        self._process_document_pages(document, document_response.full_text_annotation)
        
        return document
    
    def _process_with_document_ai(self, file_path):
        """Process document using Document AI for advanced document understanding."""
        document = DocumentStructure()
        
        # Read the file
        with open(file_path, "rb") as image_file:
            content = image_file.read()
        
        # Configure the process request
        mime_type = "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        
        # Process the document
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        result = self.docai_client.process_document(request=request)
        doc = result.document
        
        # Extract document text
        text_blocks = []
        for page in doc.pages:
            for block in page.blocks:
                # Get bounding box coordinates
                vertices = []
                for vertex in block.layout.bounding_poly.vertices:
                    vertices.append((vertex.x, vertex.y))
                
                # Get text and confidence
                block_text = self._get_text_from_layout(doc.text, block.layout)
                
                text_block = TextBlock(
                    text=block_text,
                    confidence=block.layout.confidence,
                    bounding_box=vertices
                )
                text_blocks.append(text_block)
        
        document.text_blocks = text_blocks
        
        # Extract title (if available)
        if doc.entities:
            for entity in doc.entities:
                if entity.type_ == "title":
                    document.title = entity.mention_text
        
        # Extract form fields
        form_fields = {}
        for page in doc.pages:
            for form_field in page.form_fields:
                name_text = self._get_text_from_layout(doc.text, form_field.field_name)
                value_text = self._get_text_from_layout(doc.text, form_field.field_value)
                if name_text and value_text:
                    form_fields[name_text] = value_text
        
        document.form_fields = form_fields
        
        # Extract tables
        tables = []
        for page in doc.pages:
            for table in page.tables:
                table_data = []
                for row_idx in range(table.row_span):
                    row_data = []
                    for col_idx in range(table.column_span):
                        # Find the cell at this row, column coordinate
                        cell_text = ""
                        for cell in table.body_cells:
                            if cell.row_index == row_idx and cell.column_index == col_idx:
                                cell_text = self._get_text_from_layout(doc.text, cell.layout)
                                break
                        row_data.append(cell_text)
                    table_data.append(row_data)
                
                # Get table bounding box
                vertices = []
                for vertex in table.layout.bounding_poly.vertices:
                    vertices.append((vertex.x, vertex.y))
                
                tables.append({
                    "bounding_box": vertices,
                    "data": table_data
                })
        
        document.tables = tables
        
        # Extract sections (paragraphs grouped by proximity)
        self._extract_sections_from_blocks(document)
        
        return document
    
    def _process_text_annotation(self, document, annotation):
        """Extract text blocks from Vision API full text annotation."""
        # Extract text blocks
        text_blocks = []
        for page in annotation.pages:
            for block in page.blocks:
                # Get text from block
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        block_text += word_text + " "
                
                # Get bounding box
                vertices = []
                for vertex in block.bounding_box.vertices:
                    vertices.append((vertex.x, vertex.y))
                
                # Create text block
                if block_text.strip():
                    text_block = TextBlock(
                        text=block_text.strip(),
                        confidence=block.confidence,
                        bounding_box=vertices
                    )
                    text_blocks.append(text_block)
        
        document.text_blocks = text_blocks
        
        # Try to determine document title from first text block
        if text_blocks:
            # Sort by vertical position
            sorted_blocks = sorted(text_blocks, key=lambda b: b.bounding_box[0][1])
            
            # Assume the first substantial text block might be the title
            for block in sorted_blocks[:3]:
                if len(block.text.strip()) > 3:
                    document.title = block.text.strip()
                    break
        
        return document
    
    def _extract_form_fields(self, document, annotation):
        """Extract key-value pairs that might represent form fields."""
        form_fields = {}
        
        # Look for key-value pairs in text blocks
        for block in document.text_blocks:
            text = block.text.strip()
            
            # Check for key-value pattern (Key: Value or Key - Value)
            parts = None
            if ":" in text:
                parts = text.split(":", 1)
            elif " - " in text:
                parts = text.split(" - ", 1)
            
            if parts and len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Only consider as form field if key is not too long
                if len(key) < 50:
                    form_fields[key] = value
                    block.block_type = "form_field"
                    block.metadata["key"] = key
                    block.metadata["value"] = value
        
        document.form_fields = form_fields
    
    def _process_document_pages(self, document, annotation):
        """Process document structure, identifying tables and sections."""
        # Extract tables based on layout detection
        tables = []
        
        for page in annotation.pages:
            # Look for table structures in the document
            for block in page.blocks:
                # Heuristic: blocks with multiple paragraphs in a grid-like structure might be tables
                if len(block.paragraphs) > 4:  # Potential table with multiple cells
                    # Check if paragraphs form a grid pattern
                    x_positions = set()
                    y_positions = set()
                    
                    for paragraph in block.paragraphs:
                        x = paragraph.bounding_box.vertices[0].x
                        y = paragraph.bounding_box.vertices[0].y
                        x_positions.add(x)
                        y_positions.add(y)
                    
                    # If we have multiple distinct x and y positions, it might be a table
                    if len(x_positions) > 1 and len(y_positions) > 1:
                        # Get coordinates of the table
                        vertices = []
                        for vertex in block.bounding_box.vertices:
                            vertices.append((vertex.x, vertex.y))
                        
                        # Extract table data
                        table_data = self._extract_table_data(block.paragraphs, list(y_positions), list(x_positions))
                        
                        tables.append({
                            "bounding_box": vertices,
                            "data": table_data
                        })
        
        document.tables = tables
        
        # Extract sections by grouping text blocks
        self._extract_sections_from_blocks(document)
    
    def _extract_table_data(self, paragraphs, y_positions, x_positions):
        """Extract table data from paragraphs arranged in a grid structure."""
        # Sort positions
        y_positions.sort()
        x_positions.sort()
        
        # Initialize table data
        table_data = [["" for _ in range(len(x_positions))] for _ in range(len(y_positions))]
        
        # Function to find closest position
        def find_closest_index(value, positions):
            return min(range(len(positions)), key=lambda i: abs(positions[i] - value))
        
        # Assign paragraphs to cells
        for paragraph in paragraphs:
            # Get top-left corner position
            x = paragraph.bounding_box.vertices[0].x
            y = paragraph.bounding_box.vertices[0].y
            
            # Find closest row and column
            row_idx = find_closest_index(y, y_positions)
            col_idx = find_closest_index(x, x_positions)
            
            # Get text from paragraph
            paragraph_text = ""
            for word in paragraph.words:
                word_text = "".join([symbol.text for symbol in word.symbols])
                paragraph_text += word_text + " "
            
            # Add to table data
            if 0 <= row_idx < len(y_positions) and 0 <= col_idx < len(x_positions):
                table_data[row_idx][col_idx] += paragraph_text.strip()
        
        return table_data
    
    def _extract_sections_from_blocks(self, document):
        """Group text blocks into logical sections based on formatting and position."""
        # Filter out blocks already classified as form fields or table cells
        content_blocks = [b for b in document.text_blocks 
                         if b.block_type == "text" and b.text != document.title]
        
        # Sort blocks by vertical position
        content_blocks.sort(key=lambda b: b.bounding_box[0][1])
        
        # Extract sections
        sections = []
        current_section = {"title": None, "paragraphs": []}
        
        for i, block in enumerate(content_blocks):
            text = block.text.strip()
            
            # Skip if empty
            if not text:
                continue
            
            # Heuristic to identify potential headers
            is_header = False
            if (
                len(text) < 100 and  # Not too long
                (
                    text.endswith(':') or  # Ends with colon
                    text.isupper() or  # All uppercase might be a header
                    (i > 0 and  # Has extra spacing before it
                     content_blocks[i].bounding_box[0][1] - content_blocks[i-1].bounding_box[3][1] > 20)
                )
            ):
                is_header = True
            
            if is_header:
                # Save the previous section if it has content
                if current_section["paragraphs"]:
                    sections.append(current_section)
                
                # Start a new section
                current_section = {"title": text, "paragraphs": []}
            else:
                # Add to the current section
                current_section["paragraphs"].append(text)
        
        # Add the last section
        if current_section["paragraphs"]:
            sections.append(current_section)
        
        document.sections = sections
    
    def _get_text_from_layout(self, doc_text, layout):
        """Extract text from a Document AI layout segment using text anchor information."""
        if layout.text_anchor.text_segments:
            text = ""
            for segment in layout.text_anchor.text_segments:
                start_index = segment.start_index
                end_index = segment.end_index
                text += doc_text[start_index:end_index]
            return text
        return ""


# Example usage
if __name__ == "__main__":
    # Initialize the processor using environment variables
    processor = GoogleCloudVisionProcessor()
    
    # Process a document
    document_path = r"I:\analytics_reader\ui-dashboard.png"
    result = processor.process_document(document_path)
    
    # Get JSON output
    json_output = result.to_json()
    print(json_output)
    
    # Save the JSON to a file
    with open("document_structure.json", "w") as f:
        f.write(json_output)