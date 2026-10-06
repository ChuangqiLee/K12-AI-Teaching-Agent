"""K-12 pedagogical intelligent agent (PIA).

Re-implementation of the system described in
Liu, Li, Liu, Yang, Wang & Yan (2026). Designing Multimodal Human-Computer
Interaction for K-12 Learning: Research Design, System Architecture, and Field
Evaluation of an AI Teaching Agent. International Journal of Human-Computer
Interaction. https://doi.org/10.1080/10447318.2026.2616662
"""

__version__ = "1.0.0"

from .config import load_config  # noqa: F401
from .schemas import LearnerProfile, LessonContent, LessonPackage, Question  # noqa: F401
