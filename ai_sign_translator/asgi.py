"""
ASGI config for ai_sign_translator project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_sign_translator.settings')

application = get_asgi_application()

