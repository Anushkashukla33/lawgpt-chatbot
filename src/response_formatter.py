"""
Response Formatter for Structured Output
Formats responses with bold text and proper structure as requested
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """Formats chatbot responses with structure and bold formatting"""
    
    def __init__(self):
        self.formatting_rules = {
            'bold_keywords': [
                'important', 'key', 'main', 'primary', 'essential', 'crucial',
                'definition', 'answer', 'solution', 'result', 'conclusion',
                'summary', 'overview', 'introduction', 'steps', 'process',
                'method', 'approach', 'technique', 'strategy', 'tip',
                'note', 'warning', 'caution', 'example', 'benefits',
                'advantages', 'disadvantages', 'features', 'characteristics'
            ],
            'section_headers': [
                'overview', 'introduction', 'background', 'definition',
                'explanation', 'details', 'steps', 'process', 'method',
                'examples', 'benefits', 'advantages', 'disadvantages',
                'conclusion', 'summary', 'result', 'solution'
            ]
        }
    
    def format_response(self, raw_response: str) -> str:
        """Main method to format response with structure and bold text"""
        try:
            # Clean and prepare the response
            formatted_response = self._clean_response(raw_response)
            
            # Apply structural formatting
            formatted_response = self._add_structure(formatted_response)
            
            # Apply bold formatting
            formatted_response = self._apply_bold_formatting(formatted_response)
            
            # Add section headers where appropriate
            formatted_response = self._enhance_sections(formatted_response)
            
            # Final polish
            formatted_response = self._final_polish(formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return raw_response
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize the response"""
        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in response.split('\n')]
        response = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        return response
    
    def _add_structure(self, response: str) -> str:
        """Add structural elements like bullet points and numbering"""
        lines = response.split('\n')
        structured_lines = []
        
        in_list = False
        list_items = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if in_list and list_items:
                    # End of list, add formatted list
                    structured_lines.extend(self._format_list(list_items))
                    list_items = []
                    in_list = False
                structured_lines.append('')
                continue
            
            # Check if this looks like a list item
            if self._is_list_item(line, lines, i):
                in_list = True
                list_items.append(line)
            else:
                # Not a list item
                if in_list and list_items:
                    # End previous list
                    structured_lines.extend(self._format_list(list_items))
                    list_items = []
                    in_list = False
                
                structured_lines.append(line)
        
        # Handle any remaining list items
        if in_list and list_items:
            structured_lines.extend(self._format_list(list_items))
        
        return '\n'.join(structured_lines)
    
    def _is_list_item(self, line: str, all_lines: List[str], index: int) -> bool:
        """Check if a line should be treated as a list item"""
        # Already formatted list items
        if re.match(r'^[\d\w]\.\s|^[-•*]\s', line):
            return False
        
        # Short lines that could be list items
        if len(line.split()) <= 15:
            # Check context - if previous or next line is also short, likely a list
            has_context = False
            
            if index > 0 and len(all_lines[index - 1].strip().split()) <= 15:
                has_context = True
            
            if index < len(all_lines) - 1 and len(all_lines[index + 1].strip().split()) <= 15:
                has_context = True
            
            return has_context
        
        return False
    
    def _format_list(self, items: List[str]) -> List[str]:
        """Format a list of items with proper bullet points or numbers"""
        if len(items) <= 1:
            return items
        
        formatted_items = []
        
        # Use numbers for procedural/step-like content
        if self._is_procedural_list(items):
            for i, item in enumerate(items, 1):
                formatted_items.append(f"{i}. {item}")
        else:
            # Use bullet points for general lists
            for item in items:
                formatted_items.append(f"• {item}")
        
        return formatted_items
    
    def _is_procedural_list(self, items: List[str]) -> bool:
        """Check if list items represent steps or procedures"""
        procedural_keywords = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'step', 'start', 'begin', 'create', 'setup', 'install',
            'configure', 'run', 'execute', 'complete', 'finish'
        ]
        
        procedural_count = 0
        for item in items:
            if any(keyword in item.lower() for keyword in procedural_keywords):
                procedural_count += 1
        
        return procedural_count >= len(items) * 0.5
    
    def _apply_bold_formatting(self, response: str) -> str:
        """Apply bold formatting to key terms and phrases"""
        # Don't double-bold already bolded text
        def bold_if_not_bolded(match):
            text = match.group(0)
            if '**' in text:
                return text
            return f"**{text}**"
        
        # Bold important keywords (case-insensitive)
        for keyword in self.formatting_rules['bold_keywords']:
            pattern = rf'\b{re.escape(keyword)}\b'
            response = re.sub(pattern, bold_if_not_bolded, response, flags=re.IGNORECASE)
        
        # Bold numbers when they appear to be important
        response = re.sub(r'\b(\d+(?:\.\d+)?)\s*(?:percent|%|times|fold|degrees?|years?|months?|days?|hours?|minutes?|seconds?)\b',
                         r'**\1** \2', response, flags=re.IGNORECASE)
        
        # Bold technical terms (words with specific patterns)
        response = re.sub(r'\b([A-Z]{2,})\b', r'**\1**', response)  # Acronyms
        response = re.sub(r'\b(\w+AI|\w+ML|\w+API|\w+SQL|\w+HTML|\w+CSS|\w+JS)\b', r'**\1**', response)  # Tech terms
        
        return response
    
    def _enhance_sections(self, response: str) -> str:
        """Enhance response with section headers"""
        lines = response.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Check if line should be a section header
            if self._should_be_header(line):
                # Make it a bold header
                header = line.strip()
                if not header.startswith('**'):
                    header = f"**{header.upper()}:**"
                enhanced_lines.append(header)
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _should_be_header(self, line: str) -> bool:
        """Check if a line should be treated as a section header"""
        line = line.strip().lower()
        
        # Skip if already formatted
        if '**' in line or line.startswith(('•', '-', '1.', '2.')):
            return False
        
        # Check if it's a short line with header keywords
        if len(line.split()) <= 4:
            return any(keyword in line for keyword in self.formatting_rules['section_headers'])
        
        return False
    
    def _final_polish(self, response: str) -> str:
        """Apply final polish to the formatted response"""
        # Ensure proper spacing around headers
        response = re.sub(r'(\*\*[^*]+\*\*:?)\n(?!\n)', r'\1\n\n', response)
        
        # Ensure proper spacing before lists
        response = re.sub(r'(?<!\n)\n([\d•-])', r'\n\n\1', response)
        
        # Clean up excessive spacing
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure response starts with a greeting or introduction if it's very technical
        first_line = response.split('\n')[0].strip()
        if len(first_line.split()) > 10 and not any(word in first_line.lower() for word in ['hello', 'hi', 'here', 'this', 'let me']):
            if self._is_technical_response(response):
                response = f"**Here's a comprehensive explanation:**\n\n{response}"
        
        return response.strip()
    
    def _is_technical_response(self, response: str) -> bool:
        """Check if response is technical and might benefit from an introduction"""
        technical_indicators = [
            'algorithm', 'function', 'method', 'process', 'system',
            'implementation', 'development', 'programming', 'code',
            'data', 'model', 'network', 'database', 'software'
        ]
        
        return any(indicator in response.lower() for indicator in technical_indicators)
    
    def format_error_response(self, error_message: str) -> str:
        """Format error responses consistently"""
        return f"""**⚠️ ERROR:**

{error_message}

**What you can do:**
• Try rephrasing your question
• Check for any typos in your query
• Ask a more specific question
• Contact support if the issue persists"""
    
    def format_help_response(self) -> str:
        """Format a help response"""
        return """**🤖 HOW TO USE THIS CHATBOT:**

**Getting Started:**
• Ask any question in natural language
• Be as specific as possible for better results
• Use clear and concise language

**Features:**
• **Multi-user support** - Multiple users can chat simultaneously
• **Learning capabilities** - I improve from your feedback
• **Structured responses** - Answers are formatted for clarity
• **Context awareness** - I remember our conversation

**Giving Feedback:**
• Rate responses from 1-5 stars
• Provide specific feedback to help me learn
• Your input directly improves my future responses

**Tips for Best Results:**
• Ask one question at a time
• Provide context when needed
• Use feedback features to help me learn your preferences"""
    
    def add_confidence_indicator(self, response: str, confidence: float) -> str:
        """Add confidence indicator to response"""
        if confidence >= 0.9:
            indicator = "🟢 High Confidence"
        elif confidence >= 0.7:
            indicator = "🟡 Medium Confidence"
        else:
            indicator = "🔴 Low Confidence"
        
        return f"{response}\n\n---\n*{indicator} (Score: {confidence:.2f})*"