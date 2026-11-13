"""
WSGI config for ai_sign_translator project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_sign_translator.settings')

application = get_wsgi_application()

