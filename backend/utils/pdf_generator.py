from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, darkblue
import io
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PDFGenerator:
    """Handles PDF generation from text content"""
    
    def __init__(self):
        """Initialize PDF generator with default styles"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=darkblue,
            alignment=1  # Center alignment
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=darkblue
        ))
        
        # Content style
        self.styles.add(ParagraphStyle(
            name='CustomContent',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leading=16,
            alignment=0  # Left alignment
        ))
        
        # Info style
        self.styles.add(ParagraphStyle(
            name='CustomInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=black,
            spaceAfter=6
        ))
    
    def create_pdf(self, content: str, title: str = "Video Analysis Report") -> bytes:
        """
        Create PDF from content
        
        Args:
            content: Text content to include in PDF
            title: Document title
            
        Returns:
            PDF data as bytes
        """
        try:
            # Create PDF in memory
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content
            story = []
            
            # Add title
            story.append(Paragraph(title, self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add generation info
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(f"Generated on: {generation_time}", self.styles['CustomInfo']))
            story.append(Paragraph("AI Video Analyzer - Local Processing", self.styles['CustomInfo']))
            story.append(Spacer(1, 20))
            
            # Add main content
            self._add_content_to_story(story, content)
            
            # Build PDF
            doc.build(story)
            
            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            logger.info(f"PDF generated successfully: {title}")
            return pdf_data
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            raise
    
    def _add_content_to_story(self, story: list, content: str):
        """Add formatted content to PDF story"""
        
        # Split content into sections
        sections = self._parse_content(content)
        
        for section in sections:
            if section['type'] == 'header':
                story.append(Paragraph(section['text'], self.styles['CustomSubtitle']))
                story.append(Spacer(1, 12))
            elif section['type'] == 'paragraph':
                story.append(Paragraph(section['text'], self.styles['CustomContent']))
                story.append(Spacer(1, 6))
            elif section['type'] == 'list':
                for item in section['items']:
                    story.append(Paragraph(f"• {item}", self.styles['CustomContent']))
                story.append(Spacer(1, 12))
    
    def _parse_content(self, content: str) -> list:
        """Parse content into structured sections"""
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a header (starts with keywords like "Summary:", "Key Points:", etc.)
            if any(line.lower().startswith(keyword) for keyword in ['summary:', 'key points:', 'main topics:', 'conclusion:']):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'type': 'header',
                    'text': line
                }
                sections.append(current_section)
                current_section = None
            
            # Check if line is a list item (starts with bullet points or numbers)
            elif line.startswith(('•', '-', '*')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
                if current_section and current_section['type'] != 'list':
                    sections.append(current_section)
                    current_section = None
                
                if not current_section:
                    current_section = {
                        'type': 'list',
                        'items': []
                    }
                
                # Clean up list item
                clean_item = line.lstrip('•-*0123456789.)').strip()
                current_section['items'].append(clean_item)
            
            # Regular paragraph
            else:
                if current_section and current_section['type'] != 'paragraph':
                    sections.append(current_section)
                    current_section = None
                
                if not current_section:
                    current_section = {
                        'type': 'paragraph',
                        'text': line
                    }
                else:
                    current_section['text'] += ' ' + line
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def create_transcription_pdf(self, transcription: str, video_filename: str = "Unknown") -> bytes:
        """
        Create PDF specifically for transcriptions
        
        Args:
            transcription: Video transcription text
            video_filename: Original video filename
            
        Returns:
            PDF data as bytes
        """
        try:
            buffer = io.BytesIO()
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Video Transcription", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Video info
            story.append(Paragraph(f"Video File: {video_filename}", self.styles['CustomInfo']))
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(f"Transcribed on: {generation_time}", self.styles['CustomInfo']))
            story.append(Spacer(1, 20))
            
            # Transcription content
            story.append(Paragraph("Transcription Content:", self.styles['CustomSubtitle']))
            story.append(Spacer(1, 12))
            
            # Split transcription into paragraphs for better readability
            paragraphs = transcription.split('. ')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    # Add period back if it's not the last paragraph
                    if i < len(paragraphs) - 1:
                        paragraph += '.'
                    story.append(Paragraph(paragraph.strip(), self.styles['CustomContent']))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            logger.info(f"Transcription PDF generated for: {video_filename}")
            return pdf_data
            
        except Exception as e:
            logger.error(f"Transcription PDF generation failed: {str(e)}")
            raise
    
    def create_summary_pdf(self, summary: str, transcription: str = "", video_filename: str = "Unknown") -> bytes:
        """
        Create PDF with both summary and original transcription
        
        Args:
            summary: Summary text
            transcription: Original transcription (optional)
            video_filename: Original video filename
            
        Returns:
            PDF data as bytes
        """
        try:
            buffer = io.BytesIO()
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Video Summary Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Video info
            story.append(Paragraph(f"Video File: {video_filename}", self.styles['CustomInfo']))
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(f"Generated on: {generation_time}", self.styles['CustomInfo']))
            story.append(Spacer(1, 20))
            
            # Summary section
            story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
            story.append(Spacer(1, 12))
            self._add_content_to_story(story, summary)
            story.append(Spacer(1, 20))
            
            # Full transcription (if provided)
            if transcription:
                story.append(PageBreak())
                story.append(Paragraph("Full Transcription", self.styles['CustomSubtitle']))
                story.append(Spacer(1, 12))
                
                # Split transcription into manageable chunks
                paragraphs = transcription.split('. ')
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        if i < len(paragraphs) - 1:
                            paragraph += '.'
                        story.append(Paragraph(paragraph.strip(), self.styles['CustomContent']))
                        story.append(Spacer(1, 6))
            
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            logger.info(f"Summary PDF generated for: {video_filename}")
            return pdf_data
            
        except Exception as e:
            logger.error(f"Summary PDF generation failed: {str(e)}")
            raise

    # Backward-compatible alias used by existing orchestrator code
    def generate_pdf(self, *, content: str, title: str, session_id: str) -> str:
        """Generate a PDF file on disk and return its path (legacy interface)."""
        logger.info("generate_pdf called via legacy interface; delegating to create_pdf")
        pdf_bytes = self.create_pdf(content=content, title=title)
        filename = f"{title.replace(' ', '_')}.pdf"
        return self.save_to_disk(pdf_bytes, filename)

    def save_to_disk(self, pdf_data: bytes, filename: str) -> str:
        """Persist PDF bytes under a session-aware path."""
        from pathlib import Path

        output_dir = Path("generated_pdfs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        with output_path.open('wb') as handle:
            handle.write(pdf_data)
        logger.info("PDF saved to %s", output_path)
        return str(output_path)