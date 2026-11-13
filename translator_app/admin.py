from django.contrib import admin
from .models import TranslationLog, SystemTelemetry, VocabularyItem


@admin.register(TranslationLog)
class TranslationLogAdmin(admin.ModelAdmin):
    list_display = ('predicted_token', 'confidence', 'latency_ms', 'tts_played', 'timestamp')
    list_filter = ('predicted_token', 'tts_played', 'source')
    search_fields = ('predicted_token',)
    readonly_fields = ('timestamp',)


@admin.register(SystemTelemetry)
class SystemTelemetryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'fps', 'inference_latency_ms', 'xai_latency_ms', 'total_frames_processed')


@admin.register(VocabularyItem)
class VocabularyItemAdmin(admin.ModelAdmin):
    list_display = ('token', 'category', 'benchmark_accuracy')
    search_fields = ('token', 'category')

