"""
Waiver Approvals Package
Provides human-in-the-loop staff approval workflows for fine waivers.
"""

from .waiver_approval import WaiverManager, get_waiver_manager

__all__ = ["WaiverManager", "get_waiver_manager"]
