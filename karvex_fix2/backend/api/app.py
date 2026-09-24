import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.api.models import AskRequest, AskResponse
from backend.agent import KARVEXAgent
from backend.application import create_application
from backend.ingestion.load_dataset import load_msmarco_xi_json
from backend.voice.sarvam_stt import SarvamSTT
from backend.voice.sarvam_tts import SarvamTTS


app = FastAPI(
    title="HH Goa RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# KARVEX APPLICATION
# ============================================================

application = create_application()

agent = KARVEXAgent(
    rag_service=application.service,
    model="qwen2.5:0.5b-instruct",
)


# ============================================================
# SARVAM VOICE
# ============================================================

stt = SarvamSTT()
tts = SarvamTTS()


# ============================================================
# DATASET
# ============================================================

records = load_msmarco_xi_json(
    "data/sample/dev20.json",
    limit=20,
)

application.index_records(records)


# ============================================================
# HELPERS
# ============================================================

def normalize_voice_transcript(text: str) -> str:
    """
    Correct common STT misrecognitions for the KARVEX demo.

    Sarvam can occasionally recognize:
        सौर ऊर्जा
    as:
        शौर्य ऊर्जा

    Normalize only known, high-confidence phrases.
    """

    replacements = {
        "शौर्य ऊर्जा": "सौर ऊर्जा",
        "शौर्य उर्जा": "सौर ऊर्जा",
        "सौर उर्जा": "सौर ऊर्जा",
    }

    normalized = text.strip()

    for wrong, correct in replacements.items():
        normalized = normalized.replace(
            wrong,
            correct,
        )

    return normalized


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "KARVEX",
    }


# ============================================================
# TEXT RAG / CHAT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_text(request: AskRequest):

    answer = agent.ask(
        request.query
    )

    return AskResponse(
        success=True,
        query=request.query,
        answer=answer,
        error=None,
    )


# ============================================================
# VOICE RAG
#
# Audio
#   ↓
# Sarvam STT
#   ↓
# Language detection
#   ↓
# Transcript normalization
#   ↓
# KARVEX fast voice mode
#   ↓
# Sarvam TTS
#   ↓
# WAV
# ============================================================

@app.post("/voice")
async def voice(
    file: UploadFile = File(...),
):

    voice_start = time.perf_counter()

    suffix = (
        os.path.splitext(
            file.filename or ".webm"
        )[1]
        or ".webm"
    )

    audio_bytes = await file.read()

    if not audio_bytes:
        return Response(
            content=b"",
            status_code=400,
        )

    temp_path = None

    try:

        # ====================================================
        # SAVE AUDIO
        # ====================================================

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:

            temp.write(audio_bytes)
            temp_path = temp.name

        print(
            f"[VOICE] Input audio: "
            f"{len(audio_bytes) / 1024:.1f} KB"
        )

        # ====================================================
        # SARVAM STT
        # ====================================================

        stt_start = time.perf_counter()

        with open(
            temp_path,
            "rb",
        ) as audio_file:

            stt_result = stt.transcribe(
                audio_file,
                language_code="unknown",
            )

        stt_ms = (
            time.perf_counter()
            - stt_start
        ) * 1000

        # ====================================================
        # EXTRACT TRANSCRIPT + LANGUAGE
        # ====================================================

        transcript = ""
        detected_language = "en-IN"

        if isinstance(
            stt_result,
            str,
        ):

            transcript = stt_result

            # Fallback Hindi detection
            if any(
                "\u0900" <= char <= "\u097F"
                for char in transcript
            ):
                detected_language = "hi-IN"

        else:

            transcript = getattr(
                stt_result,
                "transcript",
                "",
            )

            detected_language = (
                getattr(
                    stt_result,
                    "language_code",
                    None,
                )
                or "en-IN"
            )

        # ====================================================
        # EMPTY TRANSCRIPT
        # ====================================================

        if not transcript.strip():

            print(
                "[VOICE] Empty transcript"
            )

            return Response(
                content=b"",
                status_code=422,
            )

        # ====================================================
        # NORMALIZE VOICE TRANSCRIPT
        # ====================================================

        original_transcript = transcript

        transcript = normalize_voice_transcript(
            transcript
        )

        if transcript != original_transcript:

            print(
                "[VOICE] Transcript normalized:"
            )

            print(
                f"[VOICE] Before: "
                f"{original_transcript}"
            )

            print(
                f"[VOICE] After:  "
                f"{transcript}"
            )

        # ====================================================
        # LOG STT
        # ====================================================

        print(
            f"[VOICE] STT: "
            f"{stt_ms:.0f} ms"
        )

        print(
            f"[VOICE] Transcript: "
            f"{transcript}"
        )

        print(
            f"[VOICE] Language: "
            f"{detected_language}"
        )

        # ====================================================
        # KARVEX
        #
        # voice_mode=True activates the fast response policy:
        #
        # Normal question:
        #     short answer / lower token budget
        #
        # Explicit detailed question:
        #     longer answer allowed
        #
        # Commands:
        #     handled immediately without Ollama
        # ====================================================

        rag_start = time.perf_counter()

        answer = agent.ask(
            transcript,
            voice_mode=True,
        )

        rag_ms = (
            time.perf_counter()
            - rag_start
        ) * 1000

        print(
            f"[VOICE] RAG: "
            f"{rag_ms:.0f} ms"
        )

        print(
            f"[VOICE] Answer: "
            f"{answer}"
        )

        # ====================================================
        # TTS LANGUAGE
        # ====================================================

        supported_tts_languages = {
            "hi-IN",
            "en-IN",
            "bn-IN",
            "ta-IN",
            "te-IN",
            "gu-IN",
            "kn-IN",
            "ml-IN",
            "mr-IN",
            "pa-IN",
            "od-IN",
        }

        if (
            detected_language
            not in supported_tts_languages
        ):
            detected_language = "en-IN"

        # ====================================================
        # SARVAM TTS
        # ====================================================

        tts_start = time.perf_counter()

        audio = tts.synthesize(
            answer,
            language_code=detected_language,
        )

        tts_ms = (
            time.perf_counter()
            - tts_start
        ) * 1000

        print(
            f"[VOICE] TTS: "
            f"{tts_ms:.0f} ms"
        )

        # ====================================================
        # TOTAL
        # ====================================================

        total_ms = (
            time.perf_counter()
            - voice_start
        ) * 1000

        print(
            "========================================"
        )

        print(
            f"[VOICE] TOTAL: "
            f"{total_ms:.0f} ms"
        )

        print(
            "========================================"
        )

        # ====================================================
        # RETURN AUDIO
        #
        # IMPORTANT:
        # Do NOT put the Hindi transcript in an HTTP header.
        # Devanagari characters can cause UnicodeEncodeError.
        # ====================================================

        return Response(
            content=audio,
            media_type="audio/wav",
            headers={
                "X-KARVEX-Language": (
                    detected_language
                ),
            },
        )

    except Exception as exc:

        print(
            f"[VOICE] ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return Response(
            content=b"",
            status_code=500,
        )

    finally:

        if temp_path:

            try:
                os.remove(
                    temp_path
                )

            except OSError:
                pass