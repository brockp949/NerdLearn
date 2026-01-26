"""Test configuration"""

from apps.testing.agents.tests.example_entity_extraction_test import *
from apps.testing.agents.tests.example_follow_cable_test import *
from apps.testing.agents.tests.example_adversarial_test import *

# Mark all tests as 'agentic' for pytest filtering
pytestmark = pytest.mark.agentic
