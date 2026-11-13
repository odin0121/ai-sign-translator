"""
URL Patterns for translator_app
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('learn/', views.learning_studio, name='learning_studio'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('stream/video_feed/', views.video_feed, name='video_feed'),
    path('api/predict/', views.predict_frame, name='predict_frame'),
    path('api/speak/', views.tts_speak, name='tts_speak'),
    path('api/telemetry/', views.telemetry_stats, name='telemetry_stats'),
    path('api/vocabulary/', views.vocabulary_api, name='vocabulary_api'),
    path('api/learning/demo/', views.learning_demo, name='learning_demo'),
    path('api/clear_logs/', views.clear_logs, name='clear_logs'),
    path('api/export_csv/', views.export_csv, name='export_csv'),
    path('api/export_json/', views.export_json, name='export_json'),
    path('api/ai_chat/', views.ai_assistant_chat, name='ai_assistant_chat'),
    path('api/ai_translate/', views.ai_translate_sentence, name='ai_translate_sentence'),
]
