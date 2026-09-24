import os

from sarvamai import SarvamAI


class SarvamSTT:
    """
    Sarvam Speech-to-Text layer for KARVEX.

    Uses Saaras v3 for multilingual Indian speech.
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

    def transcribe(
        self,
        audio_file,
        language_code: str = "unknown",
    ) -> str:

        response = self.client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            language_code=language_code,
            mode="transcribe",
        )

        return response.transcript.strip()