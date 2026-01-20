"""Desktop element capture using macapptree accessibility APIs."""
from typing import Optional, Dict, Any, List
from pathlib import Path
from src.models.session_artifact import ElementInfo
from src.utils.logger import setup_logger

# Try to import macapptree, make it optional
try:
    from macapptree import get_tree, get_tree_screenshot, get_app_bundle
    MACAPPTREE_AVAILABLE = True
except ImportError:
    MACAPPTREE_AVAILABLE = False


class DesktopCapture:
    """
    Captures desktop application state using macapptree accessibility APIs.
    
    Provides:
    - Full accessibility tree of running apps
    - Element information at click coordinates
    - Bounding boxes for UI elements
    """
    
    def __init__(self):
        self.logger = setup_logger("DesktopCapture")
        
        if not MACAPPTREE_AVAILABLE:
            self.logger.warning(
                "macapptree not available. Install with: pip install macapptree\n"
                "Desktop element capture will be limited to coordinates."
            )
        
        # Cache bundle IDs
        self._bundle_cache: Dict[str, str] = {}
    
    @property
    def is_available(self) -> bool:
        """Check if desktop capture is available."""
        return MACAPPTREE_AVAILABLE
    
    def get_bundle_id(self, app_name: str) -> Optional[str]:
        """
        Get bundle ID for an application.
        
        Args:
            app_name: Application name (e.g., "Notes", "TextEdit")
        
        Returns:
            Bundle ID or None if not found
        """
        if not MACAPPTREE_AVAILABLE:
            return None
        
        # Check cache
        if app_name in self._bundle_cache:
            return self._bundle_cache[app_name]
        
        try:
            bundle = get_app_bundle(app_name)
            if bundle:
                self._bundle_cache[app_name] = bundle
                return bundle
        except Exception as e:
            self.logger.debug(f"Could not get bundle for {app_name}: {e}")
        
        return None
    
    def capture_app_tree(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Capture the full accessibility tree for an application.
        
        Args:
            app_name: Application name
        
        Returns:
            Accessibility tree as dict, or None if failed
        """
        if not MACAPPTREE_AVAILABLE:
            return None
        
        bundle = self.get_bundle_id(app_name)
        if not bundle:
            self.logger.warning(f"Could not find bundle for {app_name}")
            return None
        
        try:
            tree = get_tree(bundle)
            return tree
        except Exception as e:
            self.logger.error(f"Failed to capture tree for {app_name}: {e}")
            return None
    
    def capture_app_state(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Capture app tree and screenshots.
        
        Returns:
            Dict with tree, screenshot, and segmented screenshot
        """
        if not MACAPPTREE_AVAILABLE:
            return None
        
        bundle = self.get_bundle_id(app_name)
        if not bundle:
            return None
        
        try:
            tree, screenshot, segmented = get_tree_screenshot(bundle)
            return {
                "bundle_id": bundle,
                "tree": tree,
                "screenshot": screenshot,
                "segmented_screenshot": segmented
            }
        except Exception as e:
            self.logger.error(f"Failed to capture state for {app_name}: {e}")
            return None
    
    def find_element_at_point(
        self, 
        tree: Dict[str, Any], 
        x: int, 
        y: int
    ) -> Optional[Dict[str, Any]]:
        """
        Find the accessibility element at screen coordinates.
        
        Uses bounding box containment, returns most specific (deepest) match.
        
        Args:
            tree: Accessibility tree from macapptree
            x, y: Screen coordinates
        
        Returns:
            Element dict or None
        """
        if not tree:
            return None
        
        def point_in_bbox(node: Dict, px: int, py: int) -> bool:
            """Check if point is within node's bounding box."""
            abs_pos = node.get("absolute_position", "")
            if not abs_pos or ";" not in abs_pos:
                return False
            
            try:
                pos_parts = abs_pos.split(";")
                node_x = float(pos_parts[0])
                node_y = float(pos_parts[1])
                
                size = node.get("size", "")
                if not size or ";" not in size:
                    return False
                
                size_parts = size.split(";")
                width = float(size_parts[0])
                height = float(size_parts[1])
                
                return (node_x <= px <= node_x + width and
                        node_y <= py <= node_y + height)
            except (ValueError, IndexError):
                return False
        
        def search(node: Dict, target_x: int, target_y: int) -> Optional[Dict]:
            """Recursively search for deepest matching element."""
            if not point_in_bbox(node, target_x, target_y):
                return None
            
            # This node contains the point, check children for more specific match
            best_match = node
            
            for child in node.get("children", []):
                child_match = search(child, target_x, target_y)
                if child_match:
                    # Prefer deeper (more specific) matches
                    best_match = child_match
            
            return best_match
        
        return search(tree, x, y)
    
    def element_to_info(self, element: Dict[str, Any]) -> ElementInfo:
        """
        Convert macapptree element to ElementInfo.
        
        Args:
            element: Element dict from macapptree
        
        Returns:
            ElementInfo with accessibility data
        """
        # Parse bbox
        bbox = None
        abs_pos = element.get("absolute_position", "")
        size = element.get("size", "")
        
        if abs_pos and size and ";" in abs_pos and ";" in size:
            try:
                pos_parts = abs_pos.split(";")
                size_parts = size.split(";")
                x = int(float(pos_parts[0]))
                y = int(float(pos_parts[1]))
                w = int(float(size_parts[0]))
                h = int(float(size_parts[1]))
                bbox = [x, y, w, h]
            except (ValueError, IndexError):
                pass

        def safe_get(d, key, default=""):
            val = d.get(key)
            return val if val is not None else default
        
        return ElementInfo(
            accessibility_role=safe_get(element, "role"),
            accessibility_name=safe_get(element, "name"),
            accessibility_description=element.get("description"),
            bbox=bbox,
            absolute_position=abs_pos,
            tree_element_id=element.get("id"),
            text=element.get("value") or element.get("name"),
            role=element.get("role_description")
        )
    
    def capture_element_at_click(
        self, 
        app_name: str, 
        x: int, 
        y: int
    ) -> Optional[ElementInfo]:
        """
        Capture element information at click coordinates.
        
        Args:
            app_name: Application that was clicked
            x, y: Click coordinates
        
        Returns:
            ElementInfo or None
        """
        tree = self.capture_app_tree(app_name)
        if not tree:
            # Return coordinate-only info
            return ElementInfo(
                bbox=[x - 10, y - 10, 20, 20],
                absolute_position=f"{x};{y}"
            )
        
        element = self.find_element_at_point(tree, x, y)
        if element:
            return self.element_to_info(element)
        
        # Fallback to coordinate-only
        return ElementInfo(
            bbox=[x - 10, y - 10, 20, 20],
            absolute_position=f"{x};{y}"
        )
    
    def find_element_by_role_name(
        self, 
        tree: Dict[str, Any], 
        role: str, 
        name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find element by accessibility role and name.
        
        Args:
            tree: Accessibility tree
            role: AX role (e.g., "AXButton", "AXTextField")
            name: Element name
        
        Returns:
            Element dict or None
        """
        def search(node: Dict) -> Optional[Dict]:
            if node.get("role") == role and node.get("name") == name:
                return node
            
            for child in node.get("children", []):
                result = search(child)
                if result:
                    return result
            
            return None
        
        return search(tree)
    
    def find_elements_by_role(
        self, 
        tree: Dict[str, Any], 
        role: str
    ) -> List[Dict[str, Any]]:
        """
        Find all elements with a specific role.
        
        Args:
            tree: Accessibility tree
            role: AX role to find
        
        Returns:
            List of matching elements
        """
        results = []
        
        def search(node: Dict):
            if node.get("role") == role:
                results.append(node)
            
            for child in node.get("children", []):
                search(child)
        
        search(tree)
        return results