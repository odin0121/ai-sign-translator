"""
Dual Text-to-Speech (TTS) Engine
Integrates offline pyttsx3 and cloud gTTS with a non-blocking background queue
to maintain real-time camera inference latency <= 50ms.
"""

import os
import queue
import threading
import logging
import tempfile

logger = logging.getLogger(__name__)

# Check pyttsx3 availability
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception as e:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None
    logger.warning(f"pyttsx3 unavailable: {e}")

# Check gTTS availability
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception as e:
    GTTS_AVAILABLE = False
    gTTS = None
    logger.warning(f"gTTS unavailable: {e}")


class TTSEngine:
    """
    Non-blocking speech synthesis engine with rate limiting and duplicate suppression.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to prevent multiple audio daemon collisions
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TTSEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, speech_rate=160, volume=1.0, prefer_offline=True):
        if getattr(self, "_initialized", False):
            return

        self.speech_rate = speech_rate
        self.volume = volume
        self.prefer_offline = prefer_offline
        self.last_spoken_token = None
        self.speech_queue = queue.Queue(maxsize=10)
        
        self.pyttsx_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx_engine = pyttsx3.init()
                self.pyttsx_engine.setProperty('rate', self.speech_rate)
                self.pyttsx_engine.setProperty('volume', self.volume)
            except Exception as e:
                logger.warning(f"Could not initialize pyttsx3 engine: {e}")
                self.pyttsx_engine = None

        # Start background consumer thread
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
        self._initialized = True

    def _speech_worker(self):
        """Worker thread executing speech tasks without blocking video loop."""
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            self._synthesize_sync(text)
            self.speech_queue.task_done()

    def _synthesize_sync(self, text):
        """Synchronous internal execution of audio playback."""
        text = str(text).strip()
        if not text:
            return

        # Try pyttsx3 offline engine first if preferred and available
        if self.prefer_offline and self.pyttsx_engine is not None:
            try:
                self.pyttsx_engine.say(text)
                self.pyttsx_engine.runAndWait()
                return
            except Exception as e:
                logger.warning(f"pyttsx3 playback failed: {e}. Falling back to gTTS or system say.")

        # Try macOS native `say` utility if on Darwin
        try:
            import subprocess
            if subprocess.call(["which", "say"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                subprocess.Popen(["say", "-r", str(self.speech_rate), text])
                return
        except Exception:
            pass

        # Cloud gTTS fallback
        if GTTS_AVAILABLE and gTTS is not None:
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    temp_path = fp.name
                tts.save(temp_path)
                # Attempt playback via afplay (macOS)
                import subprocess
                subprocess.Popen(["afplay", temp_path])
            except Exception as e:
                logger.error(f"gTTS playback failed: {e}")

    def speak(self, text, allow_repeat=False):
        """
        Enqueues text for speech synthesis asynchronously.
        
        Args:
            text: phrase or sign token to pronounce
            allow_repeat: if False, skips if identical to immediate predecessor
        """
        text = str(text).strip()
        if not text:
            return False

        if not allow_repeat and text == self.last_spoken_token:
            return False

        self.last_spoken_token = text

        try:
            # Drop older pending utterances if backlog exists to maintain freshness
            if self.speech_queue.full():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    pass
            self.speech_queue.put_nowait(text)
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue speech: {e}")
            return False

    def generate_mp3(self, text, output_path):
        """
        Generates an MP3 file from text for web audio streaming endpoints.
        """
        if GTTS_AVAILABLE and gTTS is not None:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return output_path
        elif self.pyttsx_engine is not None:
            self.pyttsx_engine.save_to_file(text, output_path)
            self.pyttsx_engine.runAndWait()
            return output_path
        return None

