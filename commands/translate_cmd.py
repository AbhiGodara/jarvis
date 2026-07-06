import logging
from deep_translator import GoogleTranslator
from commands.registry import command

logger = logging.getLogger(__name__)

_LANGUAGES = {
    "hindi": "hi", "spanish": "es", "french": "fr", "german": "de",
    "arabic": "ar", "portuguese": "pt", "japanese": "ja", "chinese": "zh-CN",
    "russian": "ru", "italian": "it", "korean": "ko", "urdu": "ur",
    "bengali": "bn", "tamil": "ta", "telugu": "te", "marathi": "mr",
    "gujarati": "gu", "punjabi": "pa", "dutch": "nl", "turkish": "tr",
    "swedish": "sv", "norwegian": "no", "polish": "pl",
}


def _find_language(text: str) -> tuple[str, str] | None:
    """Detect target language from text. Returns (lang_name, lang_code) or None."""
    for lang_name, code in _LANGUAGES.items():
        if f"in {lang_name}" in text or f"to {lang_name}" in text:
            return lang_name, code
    return None


def _extract_phrase(text: str, lang_name: str) -> str:
    """Extract the phrase to be translated from the command text."""
    for trigger in ["translate", "how do you say", "say"]:
        if trigger in text:
            text = text.split(trigger, 1)[-1].strip()
            break

    for marker in [f"in {lang_name}", f"to {lang_name}", f"into {lang_name}"]:
        if marker in text:
            text = text.replace(marker, "").strip()

    return text.strip(" '\"")


@command(keywords=["translate", "how do you say", "in spanish", "in hindi", "in french",
                   "in arabic", "in german", "in japanese", "in chinese", "in russian"],
         examples=[
             "how do you say hello in spanish",
             "translate good morning to hindi",
             "what's thank you in french",
             "say i love you in japanese",
             "what would namaste be in english",
         ])
def translate(text: str) -> str:
    """Translate a phrase to another language using Google Translate."""
    lang_match = _find_language(text)

    if not lang_match:
        return "Which language would you like me to translate to? Try saying 'translate hello to Spanish'."

    lang_name, lang_code = lang_match
    phrase = _extract_phrase(text, lang_name)

    if not phrase:
        return "What phrase would you like me to translate?"

    try:
        translated = GoogleTranslator(source="auto", target=lang_code).translate(phrase)
        return f"In {lang_name}, '{phrase}' is '{translated}'."
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"I couldn't translate that to {lang_name} right now."
