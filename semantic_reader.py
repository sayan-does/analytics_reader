import os
import torch
import clip
import numpy as np
import re
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from text_layout_reader import DocumentStructure, TextBlock


@dataclass
class SemanticBlock:
    """Semantic block with CLIP embeddings and analysis"""
    source_block: TextBlock
    embedding: np.ndarray
    semantic_labels: List[str] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    category: Optional[str] = None
    importance_score: float = 0.0


@dataclass
class SemanticDocumentStructure(DocumentStructure):
    """DocumentStructure with semantic understanding"""
    semantic_blocks: List[SemanticBlock] = field(default_factory=list)
    document_categories: List[str] = field(default_factory=list)
    document_topics: List[str] = field(default_factory=list)
    key_figures: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None


class ClipSemanticProcessor:
    """Process documents using CLIP for semantic understanding"""

    def __init__(self, model_name="ViT-B/32"):
        """Initialize CLIP semantic processor"""
        # Set device and load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # Common categories and labels
        self.document_categories = [
            "invoice", "receipt", "resume", "scientific paper",
            "letter", "contract", "report", "presentation",
            "form", "manual", "brochure", "financial statement"
        ]

        self.content_type_labels = [
            "title", "heading", "paragraph", "bullet point",
            "table", "chart", "image", "signature", "date",
            "contact information", "price", "total amount"
        ]

        # Pre-encode labels for efficiency
        self.doc_category_features = self._encode_text(
            self.document_categories)
        self.content_type_features = self._encode_text(
            self.content_type_labels)

        # Regex patterns for key figures
        self.patterns = {
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b',
            'amount': r'\$\s?[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP)\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'id_number': r'\b(?:ID|No|Number|#):\s*[A-Z0-9-]+\b'
        }

    def _encode_text(self, text_list):
        """Encode text with CLIP"""
        text_tokens = clip.tokenize(text_list).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            return text_features / text_features.norm(dim=1, keepdim=True)

    def _encode_image(self, image):
        """Encode image with CLIP"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            return image_features / image_features.norm(dim=1, keepdim=True)

    def process_document(self, file_path, text_doc):
        """Process document with semantic understanding"""
        # Create semantic document structure
        semantic_doc = SemanticDocumentStructure(
            **{k: v for k, v in vars(text_doc).items()})

        try:
            # Load and process full image
            full_image = Image.open(file_path).convert("RGB")
            image_features = self._encode_image(full_image)

            # Classify document type
            similarities = 100.0 * \
                (image_features @ self.doc_category_features.T)
            probs = similarities[0].softmax(dim=0).cpu().numpy()
            top_indices = probs.argsort()[-3:][::-1]
            semantic_doc.document_categories = [
                f"{self.document_categories[i]} ({probs[i]:.2f})" for i in top_indices
            ]

            # Process each text block semantically
            semantic_blocks = []
            for block in text_doc.text_blocks:
                region = self._extract_region(full_image, block.bounding_box)
                if region and region.width >= 20 and region.height >= 20:
                    semantic_block = self._process_region(region, block)
                    semantic_blocks.append(semantic_block)

            semantic_doc.semantic_blocks = semantic_blocks

            # Extract topics, key figures, and generate summary
            semantic_doc.document_topics = self._extract_topics(
                semantic_blocks)
            semantic_doc.key_figures = self._extract_key_figures(semantic_doc)
            semantic_doc.summary = self._generate_summary(semantic_doc)

            return semantic_doc

        except Exception as e:
            raise Exception(f"Error processing document: {e}")

    def _extract_region(self, image, bbox):
        """Extract region from image using bounding box"""
        try:
            min_x = min(p[0] for p in bbox)
            min_y = min(p[1] for p in bbox)
            max_x = max(p[0] for p in bbox)
            max_y = max(p[1] for p in bbox)
            return image.crop((min_x, min_y, max_x, max_y))
        except:
            return None

    def _process_region(self, region_image, text_block):
        """Process region to extract semantic information"""
        # Encode region and classify content type
        region_features = self._encode_image(region_image)
        similarities = 100.0 * (region_features @ self.content_type_features.T)
        probs = similarities[0].softmax(dim=0).cpu().numpy()

        # Get top 3 content types
        top_indices = probs.argsort()[-3:][::-1]
        labels = [self.content_type_labels[i] for i in top_indices]
        scores = [probs[i] for i in top_indices]

        # Calculate importance score
        importance = self._calculate_importance(
            text_block, labels[0], scores[0])

        return SemanticBlock(
            source_block=text_block,
            embedding=region_features.cpu().numpy(),
            semantic_labels=labels,
            confidence_scores=scores,
            category=labels[0],
            importance_score=importance
        )

    def _calculate_importance(self, text_block, content_type, confidence):
        """Calculate importance score for a text block"""
        # Base score adjusted by content type, position and length
        score = 0.5

        # Content type adjustment
        if content_type in ["title", "heading"]:
            score += 0.3
        elif content_type in ["total amount", "signature", "date"]:
            score += 0.2
        elif content_type == "paragraph":
            score += 0.1

        # Position adjustment (top of document)
        if text_block.bounding_box[0][1] < 100:
            score += 0.1

        # Length adjustment for paragraphs
        if content_type == "paragraph" and len(text_block.text) > 200:
            score += 0.1

        # Apply confidence factor and cap at 1.0
        return min(score * confidence, 1.0)

    def _extract_topics(self, semantic_blocks):
        """Extract main document topics"""
        # First try to get topics from important headings/titles
        topics = [
            block.source_block.text for block in semantic_blocks
            if block.importance_score > 0.6 and block.category in ["title", "heading"]
        ]

        # If no topics found, use the most important blocks
        if not topics and semantic_blocks:
            topics = [
                block.source_block.text for block in
                sorted(semantic_blocks,
                       key=lambda b: b.importance_score, reverse=True)[:3]
                if block.source_block.text
            ]

        # Remove duplicates and limit to 5
        return list(dict.fromkeys(topics))[:5]

    def _extract_key_figures(self, doc):
        """Extract key figures like dates, amounts, percentages"""
        key_figures = []

        for block in doc.semantic_blocks:
            text = block.source_block.text

            # Skip if block isn't important or has no text
            if block.importance_score < 0.5 and block.category not in ['total amount', 'date', 'price']:
                continue

            # Extract figures using regex patterns
            for figure_type, pattern in self.patterns.items():
                for match in re.findall(pattern, text):
                    # Get context around the match
                    match_pos = text.find(match)
                    context = None

                    if match_pos > 0:
                        # Try to get preceding text as context
                        context_start = max(0, match_pos - 50)
                        context = text[context_start:match_pos].strip()

                        # If no context found, use following text
                        if not context:
                            context_end = min(
                                len(text), match_pos + len(match) + 50)
                            context = text[match_pos +
                                           len(match):context_end].strip()

                    # Use form field key as context if available
                    if block.source_block.block_type == "form_field" and "key" in block.source_block.metadata:
                        context = block.source_block.metadata["key"]

                    key_figures.append({
                        'type': figure_type,
                        'value': match,
                        'context': context,
                        'importance': block.importance_score
                    })

        # Sort by importance and return top 10
        return sorted(key_figures, key=lambda x: x['importance'], reverse=True)[:10]
