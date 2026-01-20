"""Gemini client wrapper - dual model approach."""
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from src.utils.logger import setup_logger
from src.utils.config import config
from src.utils.rate_limiter import rate_limiters

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class GeminiClient:
    """
    Wrapper for Google Gemini APIs.
    
    Uses TWO different models:
    - Vision model (gemini-2.0-flash): For extraction and analysis tasks
    - Computer Use model: For agentic UI control fallback
    
    Features rate limiting to prevent API quota exhaustion.
    """
    
    # Model for vision/extraction tasks (compile-time analysis)
    VISION_MODEL = "gemini-2.0-flash"
    
    # Model for agentic computer use (replay-time fallback)
    COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.google_api_key
        self.logger = setup_logger("GeminiClient")
        
        # Rate limiter for Gemini API calls
        self._rate_limiter = rate_limiters.get("gemini")
        
        if not GEMINI_AVAILABLE:
            self.logger.warning(
                "google-genai not installed. Install with: pip install google-genai\n"
                "Gemini features will be disabled."
            )
            self.client = None
        elif not self.api_key:
            self.logger.warning("No GOOGLE_API_KEY found. Gemini features disabled.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.logger.info(f"Gemini client initialized")
                self.logger.info(f"  Vision model: {self.VISION_MODEL}")
                self.logger.info(f"  Computer Use model: {self.COMPUTER_USE_MODEL}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
    
    def _acquire_rate_limit(self):
        """Acquire rate limit before making API call."""
        if self._rate_limiter:
            self._rate_limiter.acquire()
    
    @property
    def is_available(self) -> bool:
        return self.client is not None
    
    def _encode_image(self, image_path: Path) -> bytes:
        with open(image_path, "rb") as f:
            return f.read()
    
    def _safe_extract_text(self, response) -> Optional[str]:
        """Safely extract text from Gemini response."""
        try:
            if not response.candidates:
                self.logger.warning("No candidates in response")
                return None
            
            candidate = response.candidates[0]
            
            # Check for safety blocks
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == "SAFETY":
                    self.logger.warning("Response blocked by safety filter")
                    return None
            
            # Extract text from parts
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    return part.text
            
            return None
        except (AttributeError, IndexError) as e:
            self.logger.error(f"Failed to extract text: {e}")
            return None
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response, handling markdown code blocks."""
        if not text:
            return None
            
        cleaned = text.strip()
        
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Response was: {text[:500]}")
            return None
    
    # =========================================================================
    # COMPILE PHASE: Visual Analysis (uses VISION model)
    # =========================================================================
    
    def analyze_extraction_page(
        self,
        screenshot_path: Path,
        copied_value: str,
        voice_hints: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a page where user performed extraction (copy).
        
        Uses the VISION model (not Computer Use) for analysis.
        IMPORTANT: Field names should be GENERIC (e.g., "name", "address") 
        not specific to the content (e.g., NOT "the_bier_library").
        """
        if not self.is_available:
            return None
        
        voice_context = ""
        if voice_hints:
            voice_context = f"\nUser voice hints: {', '.join(voice_hints)}"
        
        prompt = f"""Analyze this screenshot where the user copied: "{copied_value}"
{voice_context}

Tasks:
1. Identify what semantic field "{copied_value}" represents
2. List ALL other extractable fields visible on this page
3. For each field, provide a visual hint for locating it
4. Classify the page type

CRITICAL RULES for field names:
- Use GENERIC field names that work for ANY entity on this type of page
- DO NOT use the actual content as field name
- Example: Use "name" or "restaurant_name", NOT "the_bier_library" or "pizza_hut"
- Example: Use "address", NOT "123_main_street"
- Example: Use "rating" or "dining_rating", NOT "4.5_stars"

Good field names: name, address, rating, rating_count, price_range, cuisine_type, phone, hours, description
Bad field names: the_bier_library, gatsby, koramangala_address, 4.5_rating

Return JSON:
{{
    "copied_field": {{
        "name": "generic_field_name_snake_case",
        "description": "What this field represents"
    }},
    "all_fields": {{
        "generic_field_name": {{
            "description": "What this field contains",
            "visual_hint": "How to visually locate it (e.g., 'large heading at top')",
            "example_value": "The actual value visible on page (for reference only)"
        }}
    }},
    "page_type": "category like restaurant_detail, search_results, product_page",
    "page_source": "website name like yelp, google, amazon"
}}

Remember: Field names must be GENERIC and reusable, not specific content values!"""

        try:
            image_bytes = self._encode_image(screenshot_path)
            
            # Rate limit before API call
            self._acquire_rate_limit()
            
            # Use VISION model for analysis (NOT Computer Use model)
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,  # <-- Key change!
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                )
            )
            
            text = self._safe_extract_text(response)
            result = self._parse_json_response(text)
            
            if result:
                self.logger.info(f"Analyzed extraction page: {len(result.get('all_fields', {}))} fields found")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to analyze extraction page: {e}")
            return None
    
    # =========================================================================
    # REPLAY PHASE: Extraction (uses VISION model)
    # =========================================================================
    
    def extract_fields(
        self,
        screenshot_bytes: bytes,
        extraction_schema: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Extract structured data from screenshot.
        
        Uses the VISION model for extraction.
        IMPORTANT: Returns data with EXACT field names from schema.
        """
        if not self.is_available:
            return None
        
        # Build the expected field names list
        expected_fields = list(extraction_schema.keys())
        
        fields_desc = []
        for field_name, field_info in extraction_schema.items():
            if isinstance(field_info, dict):
                desc = field_info.get("description", field_name)
                hint = field_info.get("visual_hint", "")
            else:
                desc = str(field_info)
                hint = ""
            if hint:
                fields_desc.append(f"- {field_name}: {desc} (look for: {hint})")
            else:
                fields_desc.append(f"- {field_name}: {desc}")
        
        fields_str = "\n".join(fields_desc)
        
        prompt = f"""Extract these fields from the screenshot:

    {fields_str}

    Return ONLY valid JSON with extracted values. 
    IMPORTANT: Use EXACTLY these field names (copy them exactly):
    {{{", ".join([f'"{f}": "value or null"' for f in expected_fields])}}}

    Rules:
    - Extract exact text as shown on page
    - Use null if field not found
    - Don't make up values
    - Field names must match EXACTLY as specified above"""

        try:
            # Rate limit before API call
            self._acquire_rate_limit()
            
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1000,
                )
            )
            
            text = self._safe_extract_text(response)
            result = self._parse_json_response(text)
            
            if result:
                # Normalize field names to match schema (handle singular/plural mismatches)
                normalized = self._normalize_field_names(result, expected_fields)
                normalized = {k: v for k, v in normalized.items() if v is not None}
                self.logger.info(f"Extracted {len(normalized)} fields")
                return normalized
            
            return result
        
        except Exception as e:
            self.logger.error(f"Field extraction failed: {e}")
            return None
    
    def extract_page_data(
        self,
        screenshot_bytes: bytes,
        context: str = ""
    ) -> Optional[Dict[str, str]]:
        """
        Extract all relevant data from a page without a predefined schema.
        
        This is useful when you don't know what fields will be on the page.
        Uses the VISION model for extraction.
        """
        if not self.is_available:
            return None
        
        context_hint = f"\nContext: The user was searching for {context}" if context else ""
        
        prompt = f"""Look at this screenshot and extract ALL relevant information visible on the page.{context_hint}

This appears to be a detail/information page. Extract:
- Name/title of the item
- Address/location if shown
- Rating(s) if shown
- Price/cost information if shown
- Phone/contact if shown
- Hours/timing if shown
- Description or category/type
- Any other key details

Return a JSON object with appropriate field names:
{{
    "name": "the main name/title",
    "address": "full address if visible",
    "rating": "rating value if shown",
    "price": "price or cost info if shown",
    "phone": "phone number if shown",
    "hours": "opening hours if shown",
    "category": "type/category",
    "description": "brief description",
    ...any other relevant fields
}}

Rules:
- Only include fields that are actually visible on the page
- Use null for fields not found
- Extract exact text as shown
- Use descriptive field names in snake_case"""

        try:
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1500,
                )
            )
            
            text = self._safe_extract_text(response)
            result = self._parse_json_response(text)
            
            if result:
                # Filter out null values
                result = {k: v for k, v in result.items() if v is not None and v != "null"}
                self.logger.info(f"Auto-extracted {len(result)} fields")
                return result
            
            return None
        
        except Exception as e:
            self.logger.error(f"Auto extraction failed: {e}")
            return None


    def _normalize_field_names(
        self, 
        extracted: Dict[str, Any], 
        expected_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Normalize extracted field names to match expected schema.
        
        Handles:
        - Singular/plural mismatches (dining_rating → dining_ratings)
        - Case differences
        - Common variations
        """
        normalized = {}
        expected_lower = {f.lower(): f for f in expected_fields}
        
        for key, value in extracted.items():
            key_lower = key.lower()
            
            # Exact match
            if key_lower in expected_lower:
                normalized[expected_lower[key_lower]] = value
                continue
            
            # Try singular/plural variations
            if key_lower.endswith('s'):
                singular = key_lower[:-1]
                if singular in expected_lower:
                    normalized[expected_lower[singular]] = value
                    continue
            else:
                plural = key_lower + 's'
                if plural in expected_lower:
                    normalized[expected_lower[plural]] = value
                    continue
            
            # Try removing/adding common suffixes
            variations = [
                key_lower.replace('_rating', '_ratings'),
                key_lower.replace('_ratings', '_rating'),
                key_lower.replace('number_of_', ''),
                'number_of_' + key_lower,
            ]
            
            matched = False
            for var in variations:
                if var in expected_lower:
                    normalized[expected_lower[var]] = value
                    matched = True
                    break
            
            if not matched:
                # Keep original if no match
                normalized[key] = value
        
        return normalized

    def validate_page_type(self, screenshot_bytes: bytes, expected_type: str) -> bool:
        """
        Verify if the page matches the expected type (e.g. "restaurant_detail").
        
        Returns:
            True if page matches expected type (or if unsure)
            False if page clearly does NOT match (e.g. asking for detail but seeing list)
        """
        if not self.is_available:
            return True
            
        prompt = f"""Look at this screenshot.
        Expected page type: "{expected_type}"

        Is this page consistent with the expected type?
        
        Examples of mismatches:
        - Expected "restaurant_detail" but see "search_results" or "list_view" -> NO
        - Expected "login_page" but see "home_page" -> NO
        
        Answer with a JSON object:
        {{
            "match": boolean,
            "actual_type": "string description of what you see"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            text = self._safe_extract_text(response)
            if not text:
                 return True
                 
            result = self._parse_json_response(text)
            if result:
                self.logger.info(f"Page validation: {result}")
                return result.get("match", True)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Page validation failed: {e}")
            return True  # Fail open
    
    def classify_page_type(self, screenshot_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Classify the type of page we're currently on.
        
        This is used for STATE AWARENESS - detecting if we're already on
        the target page type before attempting navigation.
        
        Returns:
            Dict with 'page_type' key: 'list_page', 'detail_page', 'search_results', 'home', etc.
        """
        if not self.is_available:
            return None
        
        prompt = """Classify this page. What type of page is this?

Answer with JSON:
{
    "page_type": "one of: detail_page, list_page, search_results, home_page, login_page, form_page, error_page, other",
    "is_detail_page": boolean (true if showing details of a single item like a restaurant, product, article),
    "is_list_page": boolean (true if showing multiple items/results to choose from),
    "confidence": 0.0-1.0,
    "description": "brief description of what you see"
}

Indicators of DETAIL PAGE:
- Single item with full details (name, description, reviews, images)
- Page about ONE specific restaurant, product, hotel, person, article
- Has detailed information, not just a list

Indicators of LIST PAGE:
- Multiple items/cards/results
- Grid or list of options
- Search results, category listing, product catalog"""

        try:
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                )
            )
            
            text = self._safe_extract_text(response)
            result = self._parse_json_response(text)
            
            if result:
                self.logger.debug(f"Page classification: {result.get('page_type')} (confidence: {result.get('confidence', 'N/A')})")
            
            return result
        
        except Exception as e:
            self.logger.debug(f"Page classification failed: {e}")
            return None
    
    # =========================================================================
    # REPLAY PHASE: Element Location (uses VISION model)
    # =========================================================================
    
    def find_element(
        self,
        screenshot_bytes: bytes,
        element_description: str,
        screen_width: int,
        screen_height: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find element coordinates on screen (fallback when selectors fail).
        
        Uses VISION model for element location.
        """
        if not self.is_available:
            return None
        
        prompt = f"""Find this element on the screenshot: "{element_description}"

Return JSON with the CENTER coordinates of the element.
Use coordinates relative to the image dimensions (0 to image width/height).

{{
    "found": true,
    "x": <x coordinate of center>,
    "y": <y coordinate of center>,
    "confidence": 0.9,
    "description": "what you found"
}}

If not found: {{"found": false, "reason": "explanation"}}

IMPORTANT: Return actual pixel coordinates, not normalized 0-999 values."""

        try:
            # Rate limit before API call
            self._acquire_rate_limit()
            
            response = self.client.models.generate_content(
                model=self.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                )
            )
            
            text = self._safe_extract_text(response)
            result = self._parse_json_response(text)
            
            if result and result.get("found"):
                pixel_x = int(result.get("x", 0))
                pixel_y = int(result.get("y", 0))
                
                # Clamp to screen bounds
                pixel_x = max(0, min(pixel_x, screen_width - 1))
                pixel_y = max(0, min(pixel_y, screen_height - 1))
                
                self.logger.info(f"Found element at ({pixel_x}, {pixel_y})")
                return (pixel_x, pixel_y)
            
            reason = result.get('reason', 'unknown') if result else 'no response'
            self.logger.warning(f"Element not found: {reason}")
            return None
        
        except Exception as e:
            self.logger.error(f"Element finding failed: {e}")
            return None
    
    # =========================================================================
    # REPLAY PHASE: Agentic Computer Use (uses COMPUTER USE model)
    # =========================================================================
    
    def execute_computer_use_action(
        self,
        screenshot_bytes: bytes,
        goal: str,
        screen_width: int = 1440,
        screen_height: int = 900
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single Computer Use action for a goal.
        
        This uses the COMPUTER USE model with the required tool configuration.
        
        Returns:
            Action dict like {"name": "click_at", "args": {"x": 500, "y": 300}}
            or None if no action needed / error
        """
        if not self.is_available:
            return None
        
        try:
            # Configure Computer Use tool (REQUIRED for this model)
            config = types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER
                        )
                    )
                ],
            )
            
            response = self.client.models.generate_content(
                model=self.COMPUTER_USE_MODEL,  # <-- Computer Use model
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=f"Goal: {goal}"),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=config,
            )
            
            # Extract function call from response
            candidate = response.candidates[0]
            
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    
                    # Denormalize coordinates (0-999 → actual pixels)
                    args = dict(fc.args) if fc.args else {}
                    if 'x' in args:
                        args['x'] = int(args['x'] / 1000 * screen_width)
                    if 'y' in args:
                        args['y'] = int(args['y'] / 1000 * screen_height)
                    
                    return {
                        "name": fc.name,
                        "args": args
                    }
            
            # No function call - model might have completed or given text response
            text = self._safe_extract_text(response)
            if text:
                self.logger.info(f"Computer Use response (no action): {text[:100]}")
            
            return None
        
        except Exception as e:
            self.logger.error(f"Computer Use action failed: {e}")
            return None
    
    def execute_action_loop(
        self,
        goal: str,
        get_screenshot_fn,
        execute_action_fn,
        max_iterations: int = 10
    ) -> bool:
        """
        Full agentic computer use loop.
        
        This is the proper way to use Computer Use model for multi-step tasks.
        """
        if not self.is_available:
            return False
        
        self.logger.info(f"Starting Computer Use loop for: {goal}")
        
        # Build conversation history
        contents = []
        
        # Initial request
        initial_screenshot = get_screenshot_fn()
        contents.append(
            types.Content(role="user", parts=[
                types.Part.from_text(text=goal),
                types.Part.from_bytes(data=initial_screenshot, mime_type="image/png")
            ])
        )
        
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )
            ],
        )
        
        for i in range(max_iterations):
            self.logger.debug(f"Computer Use iteration {i+1}/{max_iterations}")
            
            try:
                response = self.client.models.generate_content(
                    model=self.COMPUTER_USE_MODEL,
                    contents=contents,
                    config=config,
                )
                
                candidate = response.candidates[0]
                contents.append(candidate.content)
                
                # Check for function calls
                function_calls = [
                    part.function_call 
                    for part in candidate.content.parts 
                    if hasattr(part, 'function_call') and part.function_call
                ]
                
                if not function_calls:
                    # No actions - task complete or model gave text response
                    text = self._safe_extract_text(response)
                    self.logger.info(f"Computer Use completed: {text[:100] if text else 'No response'}")
                    return True
                
                # Execute each action
                function_responses = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}
                    
                    # Denormalize coordinates
                    # Note: execute_action_fn should handle the actual execution
                    action = {"name": fc.name, "args": args}
                    self.logger.debug(f"Executing: {action}")
                    
                    execute_action_fn(action)
                    
                    # Capture new state
                    new_screenshot = get_screenshot_fn()
                    
                    function_responses.append(
                        types.Part(function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"status": "executed"},
                            parts=[types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=new_screenshot
                                )
                            )]
                        ))
                    )
                
                # Add function responses to conversation
                contents.append(
                    types.Content(role="user", parts=function_responses)
                )
                
            except Exception as e:
                self.logger.error(f"Computer Use iteration failed: {e}")
                return False
        
        self.logger.warning(f"Computer Use hit max iterations ({max_iterations})")
        return False


# Global instance
gemini_client = GeminiClient()