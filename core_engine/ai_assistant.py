"""
AI Assistant & Linguistic Translation Engine
Provides natural language assistance, sentence restructuring, multi-language translation,
and conversational sign guidance for users.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ASL gloss abbreviations & grammar patterns to polished English
ASL_GLOSS_DICT = {
    "HELLO": "Hello",
    "THANK YOU": "Thank you very much",
    "YES": "Yes",
    "NO": "No",
    "PLEASE": "Please",
    "GOOD": "Good",
    "ME": "I",
    "YOU": "You",
    "HELP": "Help",
    "WANT": "Want",
    "NAME": "Name",
    "WHAT": "What",
    "HOW": "How",
    "WHERE": "Where",
    "WATER": "Water",
    "FOOD": "Food",
    "BATHROOM": "Bathroom",
    "DOCTOR": "Doctor",
    "EMERGENCY": "Emergency",
}

# Sign language dictionary definitions & visual guidance for assistance
SIGN_GUIDANCE_DB = {
    "A": {
        "gesture": "Closed fist with upright thumb along index finger",
        "category": "Alphabet",
        "usage": "Used for spelling names and initials (fingerspelling)",
        "tip": "Ensure your fingers are curled tight into your palm with thumb resting beside the index finger."
    },
    "B": {
        "gesture": "Open flat palm with 4 fingers pointing upright and thumb folded across palm",
        "category": "Alphabet",
        "usage": "Commonly used in signs like 'BLUE', 'BOOK', and fingerspelling",
        "tip": "Keep the 4 upright fingers pressed together straight up."
    },
    "C": {
        "gesture": "Curved fingers and thumb forming a round 'C' arc",
        "category": "Alphabet",
        "usage": "Used in 'CUP', 'CLASS', and fingerspelling",
        "tip": "Shape your whole hand like you are gripping a soft round ball."
    },
    "D": {
        "gesture": "Index finger extended straight up, middle, ring, pinky touch thumb tip",
        "category": "Alphabet",
        "usage": "Used in 'DAY', 'DATE', and fingerspelling",
        "tip": "Make sure only the index finger points up; the other 3 fingers form an 'O' ring with the thumb."
    },
    "I": {
        "gesture": "Pinky extended straight up, other 3 fingers and thumb curled into a fist",
        "category": "Alphabet",
        "usage": "Used in 'INTEREST', 'IDEA', and fingerspelling",
        "tip": "Lock the thumb over your three curled fingers."
    },
    "L": {
        "gesture": "Thumb and index finger held perpendicular forming an 'L' shape",
        "category": "Alphabet",
        "usage": "Used in 'LATER', 'LIVE', and fingerspelling",
        "tip": "Hold your thumb horizontal and index finger vertical with palm facing outward."
    },
    "V": {
        "gesture": "Index and middle fingers extended upward in a 'V' peace sign",
        "category": "Alphabet / Number 2",
        "usage": "Used in 'VICTORY', 'VISIT', 'TWO', and fingerspelling",
        "tip": "Keep ring and pinky fingers pinned down by your thumb."
    },
    "Y": {
        "gesture": "Thumb and pinky extended outward, middle 3 fingers curled into palm",
        "category": "Alphabet / Hang loose",
        "usage": "Used in 'YELLOW', 'YESTERDAY', 'PHONE', and fingerspelling",
        "tip": "Extend your thumb and pinky as far apart as comfortable."
    },
    "HELLO": {
        "gesture": "Open hand touching forehead/temple and extending outward in a salute-wave",
        "category": "Greeting",
        "usage": "Universal greeting in American Sign Language",
        "tip": "Start near your brow and smoothly push your hand outward toward the other person."
    },
    "THANK YOU": {
        "gesture": "Flat hand with fingertips touching chin, then moving forward toward the person",
        "category": "Polite Expression",
        "usage": "Expressing gratitude and politeness",
        "tip": "Do not rub; it is a gentle touch on the chin followed by an outward gesture."
    },
    "YES": {
        "gesture": "Closed fist (like an 'S' handshape) nodding up and down like a nodding head",
        "category": "Affirmation",
        "usage": "Expressing agreement or confirmation",
        "tip": "Bend your wrist forward and back two or three times."
    },
    "NO": {
        "gesture": "Index and middle fingers snap together downward to tap the thumb",
        "category": "Negation",
        "usage": "Expressing refusal or disagreement",
        "tip": "Imagine snapping a bird's beak shut with thumb, index, and middle fingers."
    },
    "PLEASE": {
        "gesture": "Open flat palm rubbing in a circular clockwise motion over the center of chest",
        "category": "Polite Expression",
        "usage": "Requesting politely",
        "tip": "Keep your palm flat against your upper chest and make two or three circles."
    }
}

# Multi-language translations for translated sentences
TRANSLATIONS = {
    "Spanish": {
        "Hello": "Hola",
        "Thank you very much": "Muchas gracias",
        "Yes": "Sí",
        "No": "No",
        "Please": "Por favor",
        "Good": "Bueno",
        "Help": "Ayuda",
        "Water": "Agua",
        "Emergency": "Emergencia",
    },
    "French": {
        "Hello": "Bonjour",
        "Thank you very much": "Merci beaucoup",
        "Yes": "Oui",
        "No": "Non",
        "Please": "S'il vous plaît",
        "Good": "Bon",
        "Help": "Aide",
        "Water": "Eau",
        "Emergency": "Urgence",
    },
    "German": {
        "Hello": "Hallo",
        "Thank you very much": "Vielen Dank",
        "Yes": "Ja",
        "No": "Nein",
        "Please": "Bitte",
        "Good": "Gut",
        "Help": "Hilfe",
        "Water": "Wasser",
        "Emergency": "Notfall",
    },
    "Hindi": {
        "Hello": "नमस्ते (Namaste)",
        "Thank you very much": "बहुत बहुत धन्यवाद (Bahut Bahut Dhanyavaad)",
        "Yes": "हाँ (Haan)",
        "No": "नहीं (Nahin)",
        "Please": "कृपया (Kripya)",
        "Good": "अच्छा (Achha)",
        "Help": "मदद (Madad)",
        "Water": "पानी (Paani)",
        "Emergency": "आपातकाल (Aapaatkaal)",
    }
}


class AIAssistantEngine:
    """
    Intelligent Assistant Engine for Sign Language Translation,
    Query Resolution, and Language Restructuring.
    """

    @classmethod
    def polish_sentence(cls, raw_tokens):
        """
        Transforms raw sign tokens (e.g. ['HELLO', 'PLEASE', 'HELP'])
        into a grammatically fluent sentence.
        """
        if not raw_tokens:
            return ""

        if isinstance(raw_tokens, str):
            words = [w.strip().upper() for w in re.split(r'[\s,]+', raw_tokens) if w.strip()]
        else:
            words = [str(w).strip().upper() for w in raw_tokens if str(w).strip()]

        if not words:
            return ""

        # Map known glosses
        polished_parts = []
        for w in words:
            if w in ASL_GLOSS_DICT:
                polished_parts.append(ASL_GLOSS_DICT[w])
            else:
                # Alphabet concatenation or proper word
                if len(w) == 1 and polished_parts and len(polished_parts[-1]) == 1:
                    polished_parts[-1] += w
                else:
                    polished_parts.append(w.capitalize())

        sentence = " ".join(polished_parts).strip()
        # Capitalize first letter and add period if not present
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()

    @classmethod
    def translate_sentence(cls, text, target_language="Spanish"):
        """Translates structured text into target language."""
        if not text:
            return ""

        if target_language == "English":
            return text

        lang_dict = TRANSLATIONS.get(target_language, {})
        # Exact match check
        cleaned = text.rstrip(".!? ").strip()
        if cleaned in lang_dict:
            return lang_dict[cleaned]

        cleaned_lower = cleaned.lower()
        for k, v in lang_dict.items():
            if k.lower() == cleaned_lower:
                return v

        # Word-by-word substitution fallback
        words = text.split()
        translated_words = []
        for w in words:
            w_clean = w.strip(".,!?")
            punct = w[len(w_clean):]
            
            matched = None
            if w_clean in lang_dict:
                matched = lang_dict[w_clean]
            else:
                for k, v in lang_dict.items():
                    if k.lower() == w_clean.lower():
                        matched = v
                        break

            if matched:
                translated_words.append(matched + punct)
            else:
                translated_words.append(w)

        return " ".join(translated_words)

    @classmethod
    def answer_query(cls, query, model_profile="wlasl100", vocabulary=None):
        """Answer common questions using verified Signa features and help content."""
        if not query:
            return (
                "Hi! I’m Signa’s built-in site assistant. Ask me how to use the camera, "
                "choose a model, practice a lesson, translate a message, or troubleshoot recognition."
            )

        q_lower = query.lower().strip()
        words = set(re.findall(r"[a-z]+", q_lower))
        model_name = {
            "isl_emergency": "ISL Emergency",
            "asl_alphabet": "ASL Alphabet · A–Z",
        }.get(model_profile, "ASL · WLASL100")

        if words.intersection({"hi", "hello", "hey"}) or any(
            phrase in q_lower for phrase in ("who are you", "what can you do")
        ):
            return (
                "Hi! I’m Signa’s built-in site assistant. I can explain the camera controls, "
                "ASL word and alphabet models, ISL model choices, supported signs, recognition "
                "confidence, Learning Studio, and message translation. I use the site’s built-in help content, so I’m focused "
                "on this app rather than general-purpose questions."
            )

        if any(phrase in q_lower for phrase in (
            "what is this site", "what does this site", "how does this site work",
            "how do i use this app", "how do i use signa", "how to use the site",
            "getting started", "get started",
        )):
            return (
                "To get started: choose ASL · WLASL100, ASL Alphabet · A–Z, or ISL Emergency at "
                "the top, allow camera access, then keep your signing hand clearly visible and "
                "well lit. The live studio "
                "shows predictions from that model’s vocabulary. Use Learn signs to watch dataset "
                "examples and practice, and use the message controls to edit, speak, or translate "
                "your text. Predictions can be wrong, so check important messages."
            )

        if words.intersection({"privacy", "private", "stored", "data", "recording"}):
            return (
                "When live recognition is running, camera frames are sent to this Signa app’s "
                "server for prediction. The built-in chat uses this app’s own Django endpoint; "
                "it does not call an external generative-AI provider. Avoid entering sensitive "
                "personal information. Data retention can depend on how this site is hosted."
            )

        if any(term in q_lower for term in ("supported signs", "supported vocabulary", "what signs", "which signs", "classes")):
            signs = vocabulary or []
            if signs:
                return (
                    f"The selected model is {model_name} and supports {len(signs)} labels: "
                        f"{', '.join(signs)}. ASL words, ASL alphabet letters, and ISL emergency "
                        "signs are separate vocabularies; switching models "
                    "changes the available labels."
                )
            return (
                f"The selected model is {model_name}. Its vocabulary is shown in the Supported "
                "Vocabulary section; word, alphabet, and ISL models use separate label sets."
            )

        if any(term in q_lower for term in ("asl vs isl", "difference between asl", "which model", "switch model", "change model")):
            return (
                "Choose the model that matches what you want to sign: ASL · WLASL100 recognizes "
                "a set of American Sign Language words, ASL Alphabet · A–Z recognizes alphabet "
                "handshapes, and ISL Emergency covers eight Indian Sign Language emergency signs. "
                "These are separate models and their vocabularies are not interchangeable. Use "
                "the selector in the top bar to switch."
            )

        if any(term in q_lower for term in ("accuracy", "accurate", "score", "performance", "confidence")):
            return (
                "Use the selected model’s held-out accuracy shown beside the model selector as a "
                "dataset benchmark, not a promise of camera accuracy. Previously measured "
                "benchmarks are 63.0% top-1 for ASL/WLASL100 (100 test clips) and 91.25% for ISL "
                "Emergency (80 signer-independent test clips). The confidence indicator can flag "
                "uncertain predictions, but it does not guarantee a prediction is correct."
            )

        if any(term in q_lower for term in ("learn", "lesson", "practice", "demonstration", "demo", "progress")):
            return (
                "Open Learn signs in the top bar. Lessons follow the selected model and play a "
                "real example from its training data. Choose a sign to watch it, then use Practice "
                "with camera or Mark as learned. Progress is saved in this browser separately for "
                "each model. Practice predictions are kept out of the translation sentence. "
                "Dataset examples are not authoritative instruction; signing varies by signer and region."
            )

        if any(term in q_lower for term in ("translate", "translation", "spanish", "french", "german", "hindi")):
            return (
                "To translate, enter or build a message in the Your Message area, choose English, "
                "Spanish, French, German, or Hindi, then select Translate. English polishes the "
                "message; the other options use the app’s built-in phrase translations, so results "
                "may be incomplete for sentences outside its supported phrases."
            )

        if any(term in q_lower for term in ("camera", "detect", "recogn", "not working", "no hand", "permission", "webcam")):
            return (
                "For camera or recognition issues: (1) allow camera permission for this site and "
                "make sure no other app is using the camera; (2) keep your hand fully in frame, "
                "centered, and well lit, with a plain background if possible; (3) select the model "
                "that matches your sign language; and (4) try a sign in that model’s Supported "
                "Vocabulary. This model recognizes only its trained labels, and predictions may "
                "still be uncertain."
            )

        if any(term in q_lower for term in ("analytics", "dashboard", "history", "logs")):
            return (
                "Open Analytics in the top bar to see recorded recognition activity and performance "
                "summaries. The dashboard reflects the data recorded by the current site deployment; "
                "it is not a live accuracy guarantee."
            )

        sign_question = re.search(
            r"\b(?:how\s+(?:(?:do|can)\s+i\s+|to\s+))?sign\s+(?:the\s+)?"
            r"(?:letter\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
            q_lower,
        )
        if sign_question:
            sign = sign_question.group(1).upper()
            info = SIGN_GUIDANCE_DB.get(sign)
            if info:
                return (
                    f"Sign guidance for {sign} ({info['category']}): {info['gesture']}. "
                    f"Usage: {info['usage']}. Tip: {info['tip']} "
                    "This is general ASL reference guidance, not a guarantee that the selected "
                    "recognition model supports the sign. Forms can vary by signer and region."
                )
            return (
                f"I don’t have step-by-step guidance for {sign} in my built-in reference. "
                "Check whether it appears in the selected model’s Supported Vocabulary or Learn "
                "signs lessons. Dataset clips are examples, not a substitute for instruction from "
                "fluent signers."
            )

        return (
            "I’m not sure about that yet. I can help with using the camera, model and vocabulary "
            "choices, recognition accuracy, Learning Studio lessons, practice progress, Analytics, "
            "and translation. Try asking about one of those features."
        )
