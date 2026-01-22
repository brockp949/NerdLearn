"""
Document Processor for Additional File Formats

Supports extraction and processing of:
- DOCX (Microsoft Word)
- PPTX (Microsoft PowerPoint)
- EPUB (Electronic Publication)
- Markdown files
- RTF (Rich Text Format)

Features:
- Text extraction with structure preservation
- Image extraction from documents
- Metadata extraction
- Table/list parsing
- Slide-by-slide processing for PPTX
"""

import io
import os
import re
import zipfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types"""
    DOCX = "docx"
    PPTX = "pptx"
    EPUB = "epub"
    MARKDOWN = "markdown"
    RTF = "rtf"
    UNKNOWN = "unknown"


@dataclass
class DocumentSection:
    """A section/chapter/slide from a document"""
    title: str
    content: str
    section_type: str  # "paragraph", "slide", "chapter", etc.
    order: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[bytes] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Result of document processing"""
    title: str
    author: Optional[str]
    document_type: DocumentType
    sections: List[DocumentSection]
    full_text: str
    word_count: int
    page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Tuple[str, bytes]] = field(default_factory=list)  # (filename, data)
    errors: List[str] = field(default_factory=list)


class BaseDocumentProcessor(ABC):
    """Base class for document processors"""

    @abstractmethod
    def can_process(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if this processor can handle the file"""
        pass

    @abstractmethod
    def process(self, file_path: str) -> ProcessedDocument:
        """Process the document and extract content"""
        pass

    @abstractmethod
    def process_bytes(self, data: bytes, filename: str) -> ProcessedDocument:
        """Process document from bytes"""
        pass


class DOCXProcessor(BaseDocumentProcessor):
    """
    Processor for Microsoft Word DOCX files.

    Extracts:
    - Text content with paragraph structure
    - Tables
    - Images
    - Document metadata
    - Styles and formatting hints
    """

    # XML namespaces used in DOCX
    NAMESPACES = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
        'dc': 'http://purl.org/dc/elements/1.1/',
    }

    def can_process(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is a DOCX"""
        if content_type:
            return content_type in [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/docx'
            ]
        return file_path.lower().endswith('.docx')

    def process(self, file_path: str) -> ProcessedDocument:
        """Process DOCX file from path"""
        with open(file_path, 'rb') as f:
            return self.process_bytes(f.read(), os.path.basename(file_path))

    def process_bytes(self, data: bytes, filename: str) -> ProcessedDocument:
        """Process DOCX from bytes"""
        sections = []
        images = []
        errors = []
        metadata = {}

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as docx:
                # Extract document.xml (main content)
                if 'word/document.xml' in docx.namelist():
                    doc_xml = docx.read('word/document.xml')
                    sections = self._extract_content(doc_xml)

                # Extract core properties (metadata)
                if 'docProps/core.xml' in docx.namelist():
                    core_xml = docx.read('docProps/core.xml')
                    metadata = self._extract_metadata(core_xml)

                # Extract images
                for name in docx.namelist():
                    if name.startswith('word/media/') and any(
                        name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']
                    ):
                        images.append((os.path.basename(name), docx.read(name)))

        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            errors.append(str(e))

        # Compile full text
        full_text = "\n\n".join(s.content for s in sections)
        word_count = len(full_text.split())

        return ProcessedDocument(
            title=metadata.get('title', filename.replace('.docx', '')),
            author=metadata.get('creator'),
            document_type=DocumentType.DOCX,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
            page_count=max(1, word_count // 500),  # Estimate
            metadata=metadata,
            images=images,
            errors=errors
        )

    def _extract_content(self, xml_data: bytes) -> List[DocumentSection]:
        """Extract content from document.xml"""
        sections = []
        root = ET.fromstring(xml_data)

        # Find all paragraphs
        paragraphs = root.findall('.//w:p', self.NAMESPACES)
        current_section_text = []
        section_order = 0

        for para in paragraphs:
            # Get all text runs in paragraph
            texts = para.findall('.//w:t', self.NAMESPACES)
            para_text = ''.join(t.text or '' for t in texts)

            # Check if this is a heading
            style = para.find('.//w:pStyle', self.NAMESPACES)
            is_heading = False
            if style is not None:
                style_val = style.get(f'{{{self.NAMESPACES["w"]}}}val', '')
                is_heading = 'Heading' in style_val or 'Title' in style_val

            if is_heading and para_text.strip():
                # Save previous section
                if current_section_text:
                    sections.append(DocumentSection(
                        title=f"Section {section_order}" if section_order > 0 else "Introduction",
                        content='\n'.join(current_section_text),
                        section_type="paragraph",
                        order=section_order
                    ))
                    section_order += 1
                    current_section_text = []

                # Start new section with heading
                current_section_text.append(f"# {para_text}")
            elif para_text.strip():
                current_section_text.append(para_text)

        # Add final section
        if current_section_text:
            sections.append(DocumentSection(
                title=f"Section {section_order}" if section_order > 0 else "Content",
                content='\n'.join(current_section_text),
                section_type="paragraph",
                order=section_order
            ))

        return sections if sections else [DocumentSection(
            title="Document",
            content="No content extracted",
            section_type="paragraph",
            order=0
        )]

    def _extract_metadata(self, xml_data: bytes) -> Dict[str, Any]:
        """Extract metadata from core.xml"""
        metadata = {}
        try:
            root = ET.fromstring(xml_data)

            # Title
            title = root.find('dc:title', self.NAMESPACES)
            if title is not None and title.text:
                metadata['title'] = title.text

            # Creator/Author
            creator = root.find('dc:creator', self.NAMESPACES)
            if creator is not None and creator.text:
                metadata['creator'] = creator.text

            # Subject
            subject = root.find('dc:subject', self.NAMESPACES)
            if subject is not None and subject.text:
                metadata['subject'] = subject.text

            # Description
            description = root.find('dc:description', self.NAMESPACES)
            if description is not None and description.text:
                metadata['description'] = description.text

        except Exception as e:
            logger.warning(f"Error extracting DOCX metadata: {e}")

        return metadata


class PPTXProcessor(BaseDocumentProcessor):
    """
    Processor for Microsoft PowerPoint PPTX files.

    Extracts:
    - Slide content (text, speaker notes)
    - Slide images
    - Presentation metadata
    - Structure (slide order, titles)
    """

    NAMESPACES = {
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    }

    def can_process(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is a PPTX"""
        if content_type:
            return content_type in [
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'application/pptx'
            ]
        return file_path.lower().endswith('.pptx')

    def process(self, file_path: str) -> ProcessedDocument:
        """Process PPTX file from path"""
        with open(file_path, 'rb') as f:
            return self.process_bytes(f.read(), os.path.basename(file_path))

    def process_bytes(self, data: bytes, filename: str) -> ProcessedDocument:
        """Process PPTX from bytes"""
        sections = []
        images = []
        errors = []
        metadata = {}

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as pptx:
                # Find all slides
                slide_files = sorted([
                    f for f in pptx.namelist()
                    if f.startswith('ppt/slides/slide') and f.endswith('.xml')
                ])

                for i, slide_file in enumerate(slide_files, 1):
                    slide_xml = pptx.read(slide_file)
                    slide_section = self._extract_slide(slide_xml, i)

                    # Try to get speaker notes
                    notes_file = f'ppt/notesSlides/notesSlide{i}.xml'
                    if notes_file in pptx.namelist():
                        notes_xml = pptx.read(notes_file)
                        notes = self._extract_text_from_xml(notes_xml)
                        if notes:
                            slide_section.metadata['speaker_notes'] = notes

                    sections.append(slide_section)

                # Extract images
                for name in pptx.namelist():
                    if name.startswith('ppt/media/') and any(
                        name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']
                    ):
                        images.append((os.path.basename(name), pptx.read(name)))

        except Exception as e:
            logger.error(f"Error processing PPTX: {e}")
            errors.append(str(e))

        full_text = "\n\n".join(
            f"Slide {s.order}: {s.title}\n{s.content}" for s in sections
        )
        word_count = len(full_text.split())

        return ProcessedDocument(
            title=metadata.get('title', filename.replace('.pptx', '')),
            author=metadata.get('creator'),
            document_type=DocumentType.PPTX,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
            page_count=len(sections),
            metadata=metadata,
            images=images,
            errors=errors
        )

    def _extract_slide(self, xml_data: bytes, slide_number: int) -> DocumentSection:
        """Extract content from a slide"""
        title = f"Slide {slide_number}"
        content_parts = []

        try:
            root = ET.fromstring(xml_data)

            # Find all text elements
            texts = root.findall('.//a:t', self.NAMESPACES)

            for i, text_elem in enumerate(texts):
                if text_elem.text:
                    # First text is usually the title
                    if i == 0:
                        title = text_elem.text
                    else:
                        content_parts.append(text_elem.text)

        except Exception as e:
            logger.warning(f"Error extracting slide {slide_number}: {e}")

        return DocumentSection(
            title=title,
            content='\n'.join(content_parts),
            section_type="slide",
            order=slide_number
        )

    def _extract_text_from_xml(self, xml_data: bytes) -> str:
        """Extract all text from XML"""
        try:
            root = ET.fromstring(xml_data)
            texts = root.findall('.//a:t', self.NAMESPACES)
            return '\n'.join(t.text for t in texts if t.text)
        except Exception:
            return ""


class EPUBProcessor(BaseDocumentProcessor):
    """
    Processor for EPUB electronic book files.

    Extracts:
    - Chapter content
    - Table of contents
    - Metadata (title, author, etc.)
    - Cover image
    """

    def can_process(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is an EPUB"""
        if content_type:
            return content_type in ['application/epub+zip', 'application/epub']
        return file_path.lower().endswith('.epub')

    def process(self, file_path: str) -> ProcessedDocument:
        """Process EPUB file from path"""
        with open(file_path, 'rb') as f:
            return self.process_bytes(f.read(), os.path.basename(file_path))

    def process_bytes(self, data: bytes, filename: str) -> ProcessedDocument:
        """Process EPUB from bytes"""
        sections = []
        images = []
        errors = []
        metadata = {}

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as epub:
                # Find the OPF file (content.opf or similar)
                opf_path = None
                for name in epub.namelist():
                    if name.endswith('.opf'):
                        opf_path = name
                        break

                if opf_path:
                    opf_content = epub.read(opf_path)
                    metadata, spine = self._parse_opf(opf_content, opf_path)

                    # Process each spine item (chapter)
                    base_path = os.path.dirname(opf_path)
                    for i, item_path in enumerate(spine, 1):
                        full_path = os.path.join(base_path, item_path) if base_path else item_path
                        # Normalize path
                        full_path = full_path.replace('\\', '/')

                        if full_path in epub.namelist():
                            html_content = epub.read(full_path)
                            section = self._process_chapter(html_content, i)
                            sections.append(section)

                # Extract cover image
                for name in epub.namelist():
                    if 'cover' in name.lower() and any(
                        name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']
                    ):
                        images.append((os.path.basename(name), epub.read(name)))
                        break

        except Exception as e:
            logger.error(f"Error processing EPUB: {e}")
            errors.append(str(e))

        full_text = "\n\n".join(
            f"Chapter {s.order}: {s.title}\n{s.content}" for s in sections
        )
        word_count = len(full_text.split())

        return ProcessedDocument(
            title=metadata.get('title', filename.replace('.epub', '')),
            author=metadata.get('creator'),
            document_type=DocumentType.EPUB,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
            page_count=len(sections),
            metadata=metadata,
            images=images,
            errors=errors
        )

    def _parse_opf(self, opf_content: bytes, opf_path: str) -> Tuple[Dict, List[str]]:
        """Parse OPF file for metadata and spine"""
        metadata = {}
        spine = []

        try:
            root = ET.fromstring(opf_content)
            ns = {
                'opf': 'http://www.idpf.org/2007/opf',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }

            # Extract metadata
            meta_elem = root.find('.//{http://www.idpf.org/2007/opf}metadata')
            if meta_elem is not None:
                title = meta_elem.find('dc:title', ns)
                if title is not None and title.text:
                    metadata['title'] = title.text

                creator = meta_elem.find('dc:creator', ns)
                if creator is not None and creator.text:
                    metadata['creator'] = creator.text

            # Build manifest lookup
            manifest = {}
            for item in root.findall('.//{http://www.idpf.org/2007/opf}item'):
                item_id = item.get('id')
                href = item.get('href')
                if item_id and href:
                    manifest[item_id] = href

            # Get spine order
            for itemref in root.findall('.//{http://www.idpf.org/2007/opf}itemref'):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    spine.append(manifest[idref])

        except Exception as e:
            logger.warning(f"Error parsing OPF: {e}")

        return metadata, spine

    def _process_chapter(self, html_content: bytes, chapter_num: int) -> DocumentSection:
        """Process HTML chapter content"""
        title = f"Chapter {chapter_num}"
        content = ""

        try:
            # Simple HTML text extraction
            text = html_content.decode('utf-8', errors='ignore')

            # Remove HTML tags
            clean_text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            clean_text = re.sub(r'<style[^>]*>.*?</style>', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            content = clean_text.strip()

            # Try to extract title from h1/h2
            title_match = re.search(r'<h[12][^>]*>([^<]+)</h[12]>', text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

        except Exception as e:
            logger.warning(f"Error processing chapter {chapter_num}: {e}")

        return DocumentSection(
            title=title,
            content=content,
            section_type="chapter",
            order=chapter_num
        )


class MarkdownProcessor(BaseDocumentProcessor):
    """
    Processor for Markdown files.

    Extracts:
    - Headers and sections
    - Code blocks
    - Links and images
    - Lists and tables
    """

    def can_process(self, file_path: str, content_type: Optional[str] = None) -> bool:
        """Check if file is Markdown"""
        if content_type:
            return content_type in ['text/markdown', 'text/x-markdown']
        return file_path.lower().endswith(('.md', '.markdown'))

    def process(self, file_path: str) -> ProcessedDocument:
        """Process Markdown file from path"""
        with open(file_path, 'rb') as f:
            return self.process_bytes(f.read(), os.path.basename(file_path))

    def process_bytes(self, data: bytes, filename: str) -> ProcessedDocument:
        """Process Markdown from bytes"""
        sections = []
        errors = []

        try:
            content = data.decode('utf-8', errors='ignore')
            sections = self._parse_markdown(content)
        except Exception as e:
            logger.error(f"Error processing Markdown: {e}")
            errors.append(str(e))

        full_text = content if 'content' in dir() else ""
        word_count = len(full_text.split())

        # Extract title from first H1
        title = filename.replace('.md', '').replace('.markdown', '')
        if sections and sections[0].title:
            title = sections[0].title

        return ProcessedDocument(
            title=title,
            author=None,
            document_type=DocumentType.MARKDOWN,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
            page_count=len(sections),
            metadata={},
            images=[],
            errors=errors
        )

    def _parse_markdown(self, content: str) -> List[DocumentSection]:
        """Parse Markdown into sections"""
        sections = []
        current_title = "Introduction"
        current_content = []
        section_order = 0

        lines = content.split('\n')

        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append(DocumentSection(
                        title=current_title,
                        content='\n'.join(current_content),
                        section_type="section",
                        order=section_order
                    ))
                    section_order += 1
                    current_content = []

                current_title = header_match.group(2).strip()
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append(DocumentSection(
                title=current_title,
                content='\n'.join(current_content),
                section_type="section",
                order=section_order
            ))

        return sections if sections else [DocumentSection(
            title="Document",
            content=content,
            section_type="section",
            order=0
        )]


class DocumentProcessorFactory:
    """
    Factory for creating appropriate document processors.
    """

    def __init__(self):
        """Initialize with all available processors"""
        self.processors = [
            DOCXProcessor(),
            PPTXProcessor(),
            EPUBProcessor(),
            MarkdownProcessor(),
        ]

    def get_processor(
        self,
        file_path: str,
        content_type: Optional[str] = None
    ) -> Optional[BaseDocumentProcessor]:
        """
        Get appropriate processor for file.

        Args:
            file_path: Path to file
            content_type: Optional MIME type

        Returns:
            Processor instance or None if unsupported
        """
        for processor in self.processors:
            if processor.can_process(file_path, content_type):
                return processor
        return None

    def process_document(
        self,
        file_path: str,
        content_type: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process document using appropriate processor.

        Args:
            file_path: Path to file
            content_type: Optional MIME type

        Returns:
            ProcessedDocument result

        Raises:
            ValueError: If file type is not supported
        """
        processor = self.get_processor(file_path, content_type)
        if not processor:
            raise ValueError(f"Unsupported document type: {file_path}")
        return processor.process(file_path)

    def process_document_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process document from bytes.

        Args:
            data: File content as bytes
            filename: Original filename
            content_type: Optional MIME type

        Returns:
            ProcessedDocument result

        Raises:
            ValueError: If file type is not supported
        """
        processor = self.get_processor(filename, content_type)
        if not processor:
            raise ValueError(f"Unsupported document type: {filename}")
        return processor.process_bytes(data, filename)

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return ['.docx', '.pptx', '.epub', '.md', '.markdown']

    def get_supported_mime_types(self) -> List[str]:
        """Get list of supported MIME types"""
        return [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/epub+zip',
            'text/markdown',
            'text/x-markdown',
        ]


# Singleton factory instance
document_processor_factory = DocumentProcessorFactory()
