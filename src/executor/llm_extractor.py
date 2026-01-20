"""LLM-based data extraction from pages."""
from pathlib import Path
from typing import Dict, Any, Optional, List
from playwright.sync_api import Page
from src.models.workflow_recipe import WorkflowStep
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client
import json


class LLMExtractor:
    """
    Extracts structured data from pages using LLM vision.
    
    Uses screenshots + page text to extract specified fields.
    """
    
    def __init__(self, page: Optional[Page] = None):
        self.page = page
        self.logger = setup_logger("LLMExtractor")
        self._extracted_data: Dict[str, Any] = {}
    
    def set_page(self, page: Page):
        """Set the browser page."""
        self.page = page
    
    @property
    def extracted_data(self) -> Dict[str, Any]:
        """Get all extracted data."""
        return self._extracted_data
    
    def extract(
        self,
        step: WorkflowStep,
        screenshot_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Extract data based on step's extraction schema.
        
        Args:
            step: Workflow step with extraction_schema
            screenshot_path: Path to screenshot (optional)
        
        Returns:
            Dict of extracted field values
        """
        schema = step.extraction_schema
        
        if not schema:
            # Default extraction fields
            schema = {
                "title": "The main title or name on the page",
                "description": "A brief description or summary",
                "key_info": "Any key information or metrics"
            }
        
        # Get page content
        page_text = self._get_page_text()
        
        # Take screenshot if not provided
        if not screenshot_path and self.page:
            screenshot_path = Path("/tmp/extraction_screenshot.png")
            try:
                self.page.screenshot(path=str(screenshot_path))
            except Exception as e:
                self.logger.warning(f"Screenshot failed: {e}")
                screenshot_path = None
        
        # Extract using LLM
        if llm_client.is_available:
            result = self._extract_with_llm(schema, page_text, screenshot_path)
        else:
            result = self._extract_with_heuristics(schema, page_text)
        
        # Store extracted data
        self._extracted_data.update(result)
        
        self.logger.info(f"Extracted {len(result)} fields")
        for field, value in result.items():
            preview = str(value)[:50] + "..." if len(str(value)) > 50 else value
            self.logger.debug(f"  {field}: {preview}")
        
        return result
    
    def _get_page_text(self) -> str:
        """Get text content from page."""
        if not self.page:
            return ""
        
        try:
            # Get main content areas
            selectors = ['main', 'article', '#content', '.content', 'body']
            
            for selector in selectors:
                try:
                    element = self.page.locator(selector).first
                    if element.count() > 0:
                        text = element.inner_text()
                        if len(text) > 100:
                            return text[:5000]
                except:
                    continue
            
            # Fallback to body
            return self.page.inner_text("body")[:5000]
        
        except Exception as e:
            self.logger.warning(f"Failed to get page text: {e}")
            return ""
    
    def _extract_with_llm(
        self,
        schema: Dict[str, str],
        page_text: str,
        screenshot_path: Optional[Path]
    ) -> Dict[str, Any]:
        """Extract using LLM vision."""
        
        schema_desc = "\n".join([f"- {k}: {v}" for k, v in schema.items()])
        
        prompt = f"""Extract specific information from this webpage.

## Fields to Extract
{schema_desc}

## Page Text Content
{page_text[:3000]}

## Instructions
1. Find the most relevant/prominent value for each field
2. Extract the exact text as shown on the page
3. If a field is not found, use null
4. Be precise - extract only what's asked for

Return JSON with extracted values:
{{{", ".join([f'"{k}": "extracted value or null"' for k in schema.keys()])}}}
"""
        
        try:
            if screenshot_path and screenshot_path.exists():
                # Use vision model
                response = llm_client.complete_with_image(prompt, screenshot_path)
            else:
                # Text-only
                response = llm_client.complete(prompt, json_response=True)
            
            if response:
                # Parse response
                result = json.loads(response) if isinstance(response, str) else response
                
                # Filter out null values
                return {k: v for k, v in result.items() if v is not None}
        
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
        
        return self._extract_with_heuristics(schema, page_text)
    
    def _extract_with_heuristics(
        self,
        schema: Dict[str, str],
        page_text: str
    ) -> Dict[str, Any]:
        """Fallback heuristic extraction."""
        result = {}
        lines = page_text.split('\n')
        
        for field, description in schema.items():
            field_lower = field.lower()
            
            # Look for common patterns
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check if line contains the field name
                if field_lower in line_lower or field_lower.replace('_', ' ') in line_lower:
                    # Try to get the value (next non-empty content)
                    for j in range(i, min(i + 3, len(lines))):
                        content = lines[j].strip()
                        if content and len(content) > 2:
                            # Clean up the content
                            if ':' in content:
                                content = content.split(':', 1)[-1].strip()
                            result[field] = content
                            break
                    break
            
            # Special handling for common fields
            if field not in result:
                if field == "title" and lines:
                    # Usually first substantial line
                    for line in lines:
                        if len(line.strip()) > 5:
                            result["title"] = line.strip()
                            break
                
                elif field in ["rating", "stars", "score"]:
                    # Look for rating patterns
                    import re
                    for line in lines:
                        match = re.search(r'(\d+\.?\d*)\s*(star|rating|/5|/10)', line.lower())
                        if match:
                            result[field] = match.group(1)
                            break
                
                elif field in ["address", "location"]:
                    # Look for address patterns
                    for line in lines:
                        if any(word in line.lower() for word in ['street', 'ave', 'road', 'blvd', 'st.']):
                            result[field] = line.strip()
                            break
        
        return result
    
    def extract_from_template(
        self,
        template: str,
        fields: List[str]
    ) -> Dict[str, str]:
        """
        Extract field values based on template and extracted data.
        
        Args:
            template: Template string with {{placeholders}}
            fields: Fields to include
        
        Returns:
            Dict mapping field names to extracted values
        """
        result = {}
        
        for field in fields:
            if field in self._extracted_data:
                result[field] = self._extracted_data[field]
        
        return result
    
    def fill_template(self, template: str) -> str:
        """
        Fill template with extracted data.
        
        Args:
            template: Template with {{field}} placeholders
        
        Returns:
            Filled template
        """
        result = template
        
        for field, value in self._extracted_data.items():
            placeholder = f"{{{{{field}}}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result