"""Element resolution with multi-strategy cascade."""
from playwright.sync_api import Page, Locator
from typing import Optional, List, Tuple
from difflib import SequenceMatcher
from src.models.element_reference import ElementReference
from src.utils.logger import setup_logger


def fuzzy_ratio(a: str, b: str) -> float:
    """Calculate fuzzy string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class ElementResolver:
    """
    Resolves ElementReferences to actual DOM elements using cascading strategies.
    
    Strategy order (by confidence):
    1. Accessibility role + label (1.0)
    2. DOM selector (0.8)
    3. Text match (variable, 0.6-0.8)
    4. Visual fallback (0.4)
    """
    
    # Confidence thresholds
    CONF_THRESH = 0.5
    TEXT_SIM_THRESH = 0.7
    
    def __init__(self, page: Page, debug: bool = False):
        """
        Initialize element resolver.
        
        Args:
            page: Playwright page object
            debug: Enable debug logging
        """
        self.page = page
        self.debug = debug
        self.logger = setup_logger("ElementResolver", structured=False)
    
    def resolve(self, ref: ElementReference) -> Optional[Locator]:
        """
        Resolve ElementReference to actual element.
        
        Args:
            ref: Element reference with multiple strategies
        
        Returns:
            Playwright Locator if found, None otherwise
        """
        candidates: List[Tuple[Locator, float]] = []
        
        if self.debug:
            self.logger.info(f"Resolving: {ref.get_description()}")
        
        # Strategy 1: Accessibility
        if ref.accessibility:
            acc_candidates = self._resolve_by_accessibility(ref)
            candidates.extend(acc_candidates)
            if self.debug and acc_candidates:
                self.logger.info(f"  Accessibility: {len(acc_candidates)} matches")
        
        # Strategy 2: DOM selector
        if ref.dom_selector:
            dom_candidate = self._resolve_by_selector(ref)
            if dom_candidate:
                candidates.append(dom_candidate)
                if self.debug:
                    self.logger.info(f"  Selector: 1 match (score: {dom_candidate[1]})")
        
        # Strategy 3: Text match
        if ref.text:
            text_candidates = self._resolve_by_text(ref)
            candidates.extend(text_candidates)
            if self.debug and text_candidates:
                self.logger.info(f"  Text: {len(text_candidates)} matches")
        
        # No candidates found
        if not candidates:
            if self.debug:
                self.logger.warning("  No candidates found")
            return None
        
        # Pick best candidate
        best_locator, best_score = max(candidates, key=lambda x: x[1])
        
        if self.debug:
            self.logger.info(f"  Best score: {best_score:.2f}")
        
        # Check threshold
        if best_score < self.CONF_THRESH:
            if self.debug:
                self.logger.warning(f"  Score below threshold ({self.CONF_THRESH})")
            return None
        
        return best_locator
    
    def _resolve_by_accessibility(self, ref: ElementReference) -> List[Tuple[Locator, float]]:
        """Resolve using accessibility role and label."""
        candidates = []
        
        if not ref.accessibility:
            return candidates
        
        acc = ref.accessibility
        
        try:
            # Try role-based selection
            if acc.role:
                # Map common roles to Playwright roles
                role_map = {
                    "button": "button",
                    "link": "link",
                    "textbox": "textbox",
                    "input": "textbox",
                    "checkbox": "checkbox",
                    "radio": "radio",
                    "heading": "heading",
                }
                
                playwright_role = role_map.get(acc.role.lower(), acc.role)
                
                # Find by role
                if acc.label:
                    # With label
                    locator = self.page.get_by_role(playwright_role, name=acc.label)
                else:
                    # Just role
                    locator = self.page.get_by_role(playwright_role)
                
                # Check if exists and visible
                if locator.count() > 0:
                    # If multiple matches, try to find visible one
                    for i in range(locator.count()):
                        elem = locator.nth(i)
                        if elem.is_visible():
                            candidates.append((elem, 1.0))
                            break
                    
                    # If no visible found, use first
                    if not candidates:
                        candidates.append((locator.first, 1.0))
        
        except Exception as e:
            pass
        
        return candidates
    
    def _resolve_by_selector(self, ref: ElementReference) -> Optional[Tuple[Locator, float]]:
        """Resolve using DOM selector."""
        if not ref.dom_selector:
            return None
        
        try:
            locator = self.page.locator(ref.dom_selector)
            
            # Check if exists and visible
            if locator.count() > 0 and locator.first.is_visible():
                return (locator.first, 0.8)
        
        except Exception:
            pass
        
        return None
    
    def _resolve_by_text(self, ref: ElementReference) -> List[Tuple[Locator, float]]:
        """Resolve using text content matching."""
        candidates = []
        
        if not ref.text or len(ref.text) < 2:
            return candidates
        
        try:
            # Common interactive elements that might contain text
            selectors = [
                "button",
                "a",
                "input[type='submit']",
                "input[type='button']",
                "[role='button']",
                "div[onclick]",
                "span[onclick]"
            ]
            
            for selector in selectors:
                elements = self.page.locator(selector).all()
                
                for elem in elements:
                    try:
                        # Get element text
                        elem_text = elem.inner_text()
                        
                        if not elem_text:
                            # Try alternative text sources
                            elem_text = elem.get_attribute("value") or ""
                            elem_text = elem_text or elem.get_attribute("aria-label") or ""
                        
                        # Calculate similarity
                        similarity = fuzzy_ratio(ref.text, elem_text)
                        
                        if similarity >= self.TEXT_SIM_THRESH:
                            score = similarity * 0.6  # Scale down text matches
                            
                            # Boost if exact match
                            if ref.text.lower() == elem_text.lower():
                                score = 0.9
                            
                            candidates.append((elem, score))
                    
                    except Exception:
                        continue
        
        except Exception:
            pass
        
        return candidates
    
    def click_element(self, ref: ElementReference, timeout: float = 5.0) -> bool:
        """
        Resolve and click an element.
        
        Args:
            ref: Element reference
            timeout: Max time to wait for element
        
        Returns:
            True if successful, False otherwise
        """
        locator = self.resolve(ref)
        
        if not locator:
            if self.debug:
                self.logger.error(f"Failed to resolve: {ref.get_description()}")
            return False
        
        try:
            locator.click(timeout=timeout * 1000)
            if self.debug:
                self.logger.info(f"Clicked: {ref.get_description()}")
            return True
        
        except Exception as e:
            if self.debug:
                self.logger.error(f"Click failed: {e}")
            return False
    
    def type_into_element(self, ref: ElementReference, text: str, timeout: float = 5.0) -> bool:
        """
        Resolve and type into an element.
        
        Args:
            ref: Element reference
            text: Text to type
            timeout: Max time to wait for element
        
        Returns:
            True if successful, False otherwise
        """
        locator = self.resolve(ref)
        
        if not locator:
            if self.debug:
                self.logger.error(f"Failed to resolve: {ref.get_description()}")
            return False
        
        try:
            locator.fill(text, timeout=timeout * 1000)
            if self.debug:
                self.logger.info(f"Typed into: {ref.get_description()}")
            return True
        
        except Exception as e:
            if self.debug:
                self.logger.error(f"Type failed: {e}")
            return False