import base64
import os

from sarvamai import SarvamAI


class SarvamTTS:
    """
    Sarvam Bulbul v3 text-to-speech for KARVEX.
    """

    def __init__(self):
        api_key = os.getenv("SARVAM_API_KEY")

        if not api_key:
            raise RuntimeError(
                "SARVAM_API_KEY environment variable is not set."
            )

        self.client = SarvamAI(
            api_subscription_key=api_key
        )

    def synthesize(
        self,
        text: str,
        language_code: str = "en-IN",
    ) -> bytes:

        if not text or not text.strip():
            raise ValueError(
                "TTS text cannot be empty."
            )

        response = self.client.text_to_speech.convert(
            text=text[:2500],
            model="bulbul:v3",
            language_code=language_code,
            speaker="shubh",
            speech_sample_rate=24000,
        )

        if not response.audios:
            raise RuntimeError(
                "Sarvam TTS returned no audio."
            )

        combined_audio = "".join(
            response.audios
        )

        return base64.b64decode(
            combined_audio
        )