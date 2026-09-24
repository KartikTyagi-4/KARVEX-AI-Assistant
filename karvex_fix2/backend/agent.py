import datetime
import difflib
import json
import os
import re
import subprocess
import webbrowser

from google import genai


class KARVEXAgent:
    """
    KARVEX desktop AI agent.

    Routes requests between:
    - fast local commands
    - Windows application control
    - websites
    - time/date
    - weather
    - document/RAG questions
    - general AI

    Important deployment behavior:
    - When running locally on Windows, application and website commands
      can launch on the local machine.
    - When running on a cloud platform such as Vercel, the server cannot
      open applications or browser tabs on the user's computer.
    """

    def __init__(
        self,
        rag_service,
        model: str = "qwen2.5:0.5b-instruct",
    ):
        self.rag_service = rag_service
        self.model = model

        self.keep_alive = -1

        # Detect whether KARVEX is running on Vercel.
        self.is_cloud = bool(
            os.environ.get("VERCEL")
            or os.environ.get("VERCEL_ENV")
        )

        # Only load Windows applications when actually running locally.
        if os.name == "nt" and not self.is_cloud:
            self.windows_apps = self._load_windows_apps()
        else:
            self.windows_apps = []

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def ask(
        self,
        query: str,
        voice_mode: bool = False,
    ) -> str:
        """
        Main KARVEX entry point.

        voice_mode=True:
            Optimizes general AI responses for spoken interaction.

        voice_mode=False:
            Normal text behavior.
        """

        query = query.strip()

        if not query:
            return "I'm listening."

        lower = self._normalize(query)

        # --------------------------------------------------------
        # IDENTITY
        # --------------------------------------------------------

        if self._contains_any(
            lower,
            [
                "who are you",
                "what is your name",
                "what\'s your name",
                "tell me your name",
                "introduce yourself",
                "who am i talking to",
                "aap kaun ho",
                "aapka naam kya hai",
                "tumhara naam kya hai",
            ],
        ):
            return "I\'m KARVEX, your intelligent AI assistant. How can I help you today?"

        # --------------------------------------------------------
        # OPEN / LAUNCH / START
        # --------------------------------------------------------

        if self._is_open_command(lower):
            return self._handle_open_command(query)

        # --------------------------------------------------------
        # CLOSE / EXIT / QUIT
        # --------------------------------------------------------

        if self._is_close_command(lower):
            return self._handle_close_command(query)

        # --------------------------------------------------------
        # TIME
        # --------------------------------------------------------

        if self._contains_any(
            lower,
            [
                "what time is it",
                "what's the time",
                "what is the time",
                "current time",
                "tell me the time",
                "time right now",
                "what time right now",
                "समय क्या है",
                "अभी कितने बजे हैं",
                "कितने बजे हैं",
            ],
        ):
            now = datetime.datetime.now()

            return (
                f"It is "
                f"{now.strftime('%I:%M %p').lstrip('0')}."
            )

        # --------------------------------------------------------
        # DATE
        # --------------------------------------------------------

        if self._contains_any(
            lower,
            [
                "what date is it",
                "today's date",
                "what is today's date",
                "current date",
                "what day is it",
                "what is today",
                "आज की तारीख",
                "आज कौन सा दिन है",
            ],
        ):
            now = datetime.datetime.now()

            return (
                f"Today is "
                f"{now.strftime('%A, %B %d, %Y')}."
            )

        # --------------------------------------------------------
        # WEATHER
        # --------------------------------------------------------

        if self._contains_any(
            lower,
            [
                "weather",
                "temperature outside",
                "temperature today",
                "is it raining",
                "will it rain",
                "forecast",
                "how's the weather",
                "how is the weather",
                "मौसम",
                "आज का मौसम",
                "बारिश",
                "तापमान",
            ],
        ):
            return self._get_weather()

        # --------------------------------------------------------
        # DOCUMENT / RAG
        # --------------------------------------------------------

        if self._looks_like_document_question(lower):
            try:
                result = self.rag_service.ask(query)

                if result and getattr(result, "success", False):
                    answer = getattr(
                        result,
                        "answer",
                        "",
                    )

                    if answer and answer.strip():
                        return answer.strip()

            except Exception as exc:
                print(f"KARVEX RAG error: {exc}")

        # --------------------------------------------------------
        # GENERAL AI
        # --------------------------------------------------------

        return self._general_ai(
            query,
            voice_mode=voice_mode,
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize commands and speech-transcribed input.
        """

        text = text.lower().strip()

        text = re.sub(
            r"[?!.,]+$",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # ============================================================
    # COMMAND DETECTION
    # ============================================================

    @staticmethod
    def _is_open_command(query: str) -> bool:
        prefixes = (
            "open ",
            "launch ",
            "start ",
            "run ",
            "खोलो ",
            "ओपन ",
            "चलाओ ",
            "चला दो ",
        )

        return query.startswith(prefixes)

    @staticmethod
    def _is_close_command(query: str) -> bool:
        prefixes = (
            "close ",
            "exit ",
            "quit ",
            "stop ",
            "बंद करो ",
            "बंद ",
            "क्लोज ",
        )

        return query.startswith(prefixes)

    # ============================================================
    # RESPONSE LENGTH
    # ============================================================

    @staticmethod
    def _wants_detailed_answer(query: str) -> bool:
        """
        Detect explicit requests for a detailed explanation.
        """

        text = query.lower().strip()

        detail_phrases = (
            "explain in detail",
            "explain this in detail",
            "explain thoroughly",
            "explain deeply",
            "give me details",
            "give me more details",
            "give me a detailed explanation",
            "give me a detailed answer",
            "give me a long answer",
            "give me a full explanation",
            "give me the full explanation",
            "tell me everything",
            "in detail",
            "in depth",
            "deep dive",
            "detailed explanation",
            "detailed answer",
            "long answer",
            "thorough explanation",
            "elaborate",
            "go into detail",
            "explain everything",
            "विस्तार से बताओ",
            "विस्तार से समझाओ",
            "पूरी जानकारी दो",
            "डिटेल में बताओ",
            "विस्तार में बताओ",
            "विस्तार से समझाइए",
            "पूरी जानकारी चाहिए",
            "सब कुछ बताओ",
        )

        return any(
            phrase in text
            for phrase in detail_phrases
        )

    # ============================================================
    # GENERAL AI
    # ============================================================

    def _general_ai(
        self,
        query: str,
        voice_mode: bool = False,
    ) -> str:
        """
        Generate a general AI response using Gemini.
        """

        wants_detail = self._wants_detailed_answer(query)

        # --------------------------------------------------------
        # Response policy
        # --------------------------------------------------------

        if voice_mode and not wants_detail:
            response_instruction = """
- This is a voice conversation.
- Answer naturally and directly.
- Normally use 2 to 4 short sentences.
- Prefer approximately 30 to 80 words.
- Do not repeat the user's question.
- Do not add unnecessary filler.
"""

            max_tokens = 128

        elif wants_detail:
            response_instruction = """
- The user explicitly wants a detailed explanation.
- Give a complete and useful answer.
- Use multiple paragraphs or bullet points when helpful.
- Explain the important concepts clearly.
- Do not artificially shorten the response.
"""

            max_tokens = 512

        else:
            response_instruction = """
- Give a useful and complete answer.
- Answer the user's actual question directly.
- Use enough explanation to make the answer useful.
- Do not artificially shorten your response.
- Normally provide at least one complete explanation rather than
  only a fragment or partial sentence.
"""

            max_tokens = 384

        # --------------------------------------------------------
        # System prompt
        # --------------------------------------------------------

        system_prompt = f"""
You are KARVEX, an intelligent AI assistant.

You are speaking with the user naturally.

LANGUAGE:
- Respond in the same language as the user.
- English -> English.
- Hindi -> Hindi.
- Hinglish -> natural Hinglish.
- Bhojpuri -> Bhojpuri when possible.
- Haryanavi -> Haryanavi when possible.
- Do not unnecessarily translate the user's language.
- If the user mixes Hindi and English, naturally mix them.

RESPONSE LENGTH:
{response_instruction}

STYLE:
- Sound intelligent, natural, friendly, and confident.
- Use simple language unless technical detail is requested.
- Give the answer first.
- Avoid unnecessary filler.
- Do not intentionally stop in the middle of a sentence.
- Complete your answer before finishing.

IMPORTANT:
- Answer normal general-knowledge questions directly.
- Do not invent facts when uncertain.
- Do not mention internal implementation details unless asked.
- Do not claim that you opened an application or controlled a
  computer unless that action was actually performed locally.
"""

        try:
            api_key = os.environ.get(
                "GEMINI_API_KEY"
            )

            if not api_key:
                print(
                    "KARVEX Gemini error: "
                    "GEMINI_API_KEY is not configured."
                )

                return (
                    "The AI service is not configured right now."
                )

            # Allow changing the model through an environment variable.
            gemini_model = os.environ.get(
                "GEMINI_MODEL",
                "gemini-3.6-flash",
            )

            client = genai.Client(
                api_key=api_key
            )

            response = client.models.generate_content(
                model=gemini_model,
                contents=query,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=max_tokens,
                ),
            )

            # ----------------------------------------------------
            # Safely extract response text.
            # ----------------------------------------------------

            answer = ""

            try:
                answer = (
                    getattr(response, "text", "")
                    or ""
                ).strip()

            except Exception:
                answer = ""

            # Fallback extraction from candidates.
            if not answer:
                try:
                    candidates = getattr(
                        response,
                        "candidates",
                        [],
                    )

                    if candidates:
                        candidate = candidates[0]

                        content = getattr(
                            candidate,
                            "content",
                            None,
                        )

                        parts = getattr(
                            content,
                            "parts",
                            [],
                        )

                        texts = []

                        for part in parts:
                            text = getattr(
                                part,
                                "text",
                                None,
                            )

                            if text:
                                texts.append(text)

                        answer = "".join(
                            texts
                        ).strip()

                except Exception as extraction_error:
                    print(
                        "KARVEX response extraction error:",
                        extraction_error,
                    )

            if not answer:
                print(
                    "KARVEX Gemini returned an empty response."
                )

                return (
                    "I couldn't generate a response for that right now."
                )

            return answer

        except Exception as exc:
            error_message = str(exc)

            print(
                f"KARVEX Gemini error: "
                f"{error_message}"
            )

            # ----------------------------------------------------
            # Handle quota exhaustion clearly.
            # ----------------------------------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "quota" in error_message.lower()
            ):
                return (
                    "The AI service has temporarily reached its "
                    "request limit. Please try again in a minute."
                )

            return (
                "I couldn't process that request right now. "
                "Please try again."
            )

    # ============================================================
    # DOCUMENT ROUTING
    # ============================================================

    def _looks_like_document_question(
        self,
        query: str,
    ) -> bool:
        """
        Only explicitly document-related questions go through RAG.
        """

        document_terms = [
            "according to the document",
            "according to the documents",
            "according to the file",
            "according to the files",
            "in the document",
            "in the documents",
            "in the file",
            "in the files",
            "from the document",
            "from the documents",
            "from the file",
            "from the files",
            "uploaded document",
            "uploaded file",
            "provided document",
            "provided file",
            "what does the document say",
            "what do the documents say",
            "summarize the document",
            "summarize the documents",
            "summarize the file",
            "summarize the files",
            "based on the document",
            "based on the documents",
            "based on the file",
            "based on the files",
        ]

        return any(
            term in query
            for term in document_terms
        )

    # ============================================================
    # WINDOWS START APPS
    # ============================================================

    def _load_windows_apps(self):
        """
        Read installed Windows Start Menu applications.
        """

        if os.name != "nt":
            return []

        try:
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "Get-StartApps | "
                    "Select-Object Name,AppID | "
                    "ConvertTo-Json -Compress"
                ),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
            )

            if result.returncode != 0:
                print(
                    "Could not read Windows Start Apps:",
                    result.stderr,
                )

                return []

            raw = result.stdout.strip()

            if not raw:
                return []

            data = json.loads(raw)

            if isinstance(data, dict):
                data = [data]

            apps = []

            for item in data:
                name = str(
                    item.get("Name", "")
                ).strip()

                app_id = str(
                    item.get("AppID", "")
                ).strip()

                if name and app_id:
                    apps.append(
                        {
                            "name": name,
                            "app_id": app_id,
                        }
                    )

            print(
                f"KARVEX loaded "
                f"{len(apps)} Windows apps."
            )

            return apps

        except Exception as exc:
            print(
                f"Windows app discovery error: {exc}"
            )

            return []

    # ============================================================
    # OPEN APPLICATIONS / WEBSITES
    # ============================================================

    def _handle_open_command(
        self,
        query: str,
    ) -> str:
        """
        Handle application and website opening commands.
        """

        target = self._normalize(query)

        target = re.sub(
            r"^(open|launch|start|run)\s+",
            "",
            target,
        )

        target = re.sub(
            r"^(खोलो|ओपन|चलाओ|चला दो)\s+",
            "",
            target,
        )

        target = target.strip()

        if not target:
            return "What would you like me to open?"

        websites = {
            "gmail": "https://mail.google.com",
            "google mail": "https://mail.google.com",
            "youtube": "https://www.youtube.com",
            "github": "https://github.com",
            "chatgpt": "https://chatgpt.com",
            "linkedin": "https://www.linkedin.com",
            "linkedin web": "https://www.linkedin.com",
            "linkedin website": "https://www.linkedin.com",
            "whatsapp": "https://web.whatsapp.com",
            "whatsapp web": "https://web.whatsapp.com",
            "notion": "https://www.notion.so",
            "notion web": "https://www.notion.so",
            "google": "https://www.google.com",
            "google search": "https://www.google.com",
            "facebook": "https://www.facebook.com",
            "instagram": "https://www.instagram.com",
            "twitter": "https://x.com",
            "x": "https://x.com",
        }

        # --------------------------------------------------------
        # Website aliases
        # --------------------------------------------------------

        if target in websites:

            # A cloud server cannot open a browser on the user's PC.
            if self.is_cloud:
                return (
                    f"{target.title()} cannot be opened directly "
                    f"on your computer from the deployed cloud version. "
                    f"Run KARVEX locally to launch it automatically."
                )

            try:
                opened = webbrowser.open(
                    websites[target]
                )

                if opened:
                    return f"Opening {target}."

                return (
                    f"I tried to open {target}, "
                    f"but the browser did not confirm the action."
                )

            except Exception as exc:
                print(
                    f"Website open error: {exc}"
                )

                return (
                    f"I couldn't open {target}."
                )

        # --------------------------------------------------------
        # Cloud limitation
        # --------------------------------------------------------

        if self.is_cloud:
            return (
                f"I can't open {target} on your physical computer "
                f"from the Vercel deployment. Application control "
                f"requires KARVEX to run locally on your Windows PC."
            )

        # --------------------------------------------------------
        # Common application aliases
        # --------------------------------------------------------

        aliases = {
            "chrome": [
                "google chrome",
                "chrome",
            ],
            "google chrome": [
                "google chrome",
                "chrome",
            ],
            "brave": [
                "brave",
            ],
            "linkedin": [
                "linkedin",
            ],
            "whatsapp": [
                "whatsapp",
            ],
            "vs code": [
                "visual studio code",
                "vs code",
            ],
            "visual studio code": [
                "visual studio code",
                "vs code",
            ],
            "code": [
                "visual studio code",
            ],
            "cursor": [
                "cursor",
            ],
            "discord": [
                "discord",
            ],
            "zoom": [
                "zoom workplace",
                "zoom",
            ],
            "excel": [
                "excel",
            ],
            "word": [
                "word",
            ],
            "powerpoint": [
                "powerpoint",
            ],
            "power point": [
                "powerpoint",
            ],
            "outlook": [
                "outlook",
            ],
            "notepad": [
                "notepad",
            ],
            "calculator": [
                "calculator",
            ],
            "calc": [
                "calculator",
            ],
            "paint": [
                "paint",
            ],
            "file explorer": [
                "file explorer",
            ],
            "explorer": [
                "file explorer",
            ],
            "task manager": [
                "task manager",
            ],
            "settings": [
                "settings",
            ],
            "obsidian": [
                "obsidian",
            ],
            "notion": [
                "notion",
            ],
            "teams": [
                "microsoft teams",
                "teams",
            ],
            "onedrive": [
                "onedrive",
            ],
            "onenote": [
                "onenote",
            ],
            "power automate": [
                "power automate",
            ],
        }

        search_terms = [target]

        if target in aliases:
            search_terms.extend(
                aliases[target]
            )

        simplified = re.sub(
            r"\b(app|application|program|browser)\b",
            "",
            target,
        ).strip()

        if simplified:
            search_terms.append(
                simplified
            )

        # --------------------------------------------------------
        # Find installed Windows app
        # --------------------------------------------------------

        app = self._find_windows_app(
            search_terms
        )

        if app:
            return self._launch_windows_app(
                app
            )

        # --------------------------------------------------------
        # Built-in Windows programs
        # --------------------------------------------------------

        builtins = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "paint": "mspaint.exe",
            "command prompt": "cmd.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "file explorer": "explorer.exe",
            "explorer": "explorer.exe",
            "task manager": "taskmgr.exe",
            "settings": "ms-settings:",
        }

        if target in builtins:
            command = builtins[target]

            try:
                if command.endswith(":"):
                    os.startfile(command)
                else:
                    subprocess.Popen(
                        command,
                        shell=False,
                    )

                return f"Opening {target}."

            except Exception as exc:
                print(
                    f"Built-in launch error: {exc}"
                )

                return (
                    f"I couldn't open {target}."
                )

        # --------------------------------------------------------
        # Direct URL
        # --------------------------------------------------------

        if (
            target.startswith("http://")
            or target.startswith("https://")
        ):
            try:
                opened = webbrowser.open(target)

                if opened:
                    return "Opening it."

                return (
                    "I tried to open the website, "
                    "but the browser did not confirm the action."
                )

            except Exception as exc:
                print(
                    f"Direct website error: {exc}"
                )

                return (
                    "I couldn't open that website."
                )

        return (
            f"I couldn't find an installed app called "
            f"{target}."
        )

    # ============================================================
    # FIND WINDOWS APP
    # ============================================================

    def _find_windows_app(
        self,
        search_terms,
    ):
        """
        Find the best matching Windows application.

        Matching order:
        1. Exact
        2. Starts-with
        3. Contains
        4. Fuzzy similarity
        """

        if not self.windows_apps:
            return None

        normalized_apps = []

        for app in self.windows_apps:
            name = self._normalize(
                app["name"]
            )

            normalized_apps.append(
                (
                    name,
                    app,
                )
            )

        # Exact match
        for term in search_terms:
            term = self._normalize(term)

            for name, app in normalized_apps:
                if name == term:
                    return app

        # Starts-with match
        for term in search_terms:
            term = self._normalize(term)

            candidates = [
                app
                for name, app in normalized_apps
                if name.startswith(term)
            ]

            if candidates:
                return min(
                    candidates,
                    key=lambda item: len(
                        item["name"]
                    ),
                )

        # Contains match
        for term in search_terms:
            term = self._normalize(term)

            candidates = [
                app
                for name, app in normalized_apps
                if term in name
            ]

            if candidates:
                return min(
                    candidates,
                    key=lambda item: len(
                        item["name"]
                    ),
                )

        # Fuzzy match
        names = [
            name
            for name, _ in normalized_apps
        ]

        for term in search_terms:
            term = self._normalize(term)

            if len(term) < 3:
                continue

            matches = difflib.get_close_matches(
                term,
                names,
                n=1,
                cutoff=0.72,
            )

            if matches:
                best_name = matches[0]

                for name, app in normalized_apps:
                    if name == best_name:
                        return app

        return None

    # ============================================================
    # LAUNCH WINDOWS APP
    # ============================================================

    def _launch_windows_app(
        self,
        app,
    ) -> str:
        """
        Launch a Windows Start App through its AppID.
        """

        if os.name != "nt":
            return (
                "Windows application launching is only available "
                "when KARVEX is running locally on Windows."
            )

        name = app["name"]
        app_id = app["app_id"]

        try:
            subprocess.Popen(
                [
                    "explorer.exe",
                    f"shell:AppsFolder\\{app_id}",
                ],
                shell=False,
            )

            return f"Opening {name}."

        except Exception as exc:
            print(
                f"Windows app launch error for "
                f"{name}: {exc}"
            )

            try:
                os.startfile(
                    f"shell:AppsFolder\\{app_id}"
                )

                return f"Opening {name}."

            except Exception as fallback_exc:
                print(
                    "Fallback launch failed:",
                    fallback_exc,
                )

        return (
            f"I found {name}, but Windows "
            f"couldn't launch it."
        )

    # ============================================================
    # CLOSE APPLICATIONS
    # ============================================================

    def _handle_close_command(
        self,
        query: str,
    ) -> str:
        """
        Safely close selected applications.
        """

        target = self._normalize(query)

        target = re.sub(
            r"^(close|exit|quit|stop)\s+",
            "",
            target,
        )

        target = re.sub(
            r"^(बंद करो|बंद|क्लोज)\s+",
            "",
            target,
        )

        target = target.strip()

        if self.is_cloud:
            return (
                f"I can't close {target} on your physical computer "
                f"from the cloud deployment. Run KARVEX locally "
                f"for computer control."
            )

        processes = {
            "notepad": [
                "notepad.exe",
            ],
            "calculator": [
                "CalculatorApp.exe",
            ],
            "calc": [
                "CalculatorApp.exe",
            ],
            "paint": [
                "mspaint.exe",
            ],
            "whatsapp": [
                "WhatsApp.exe",
            ],
            "chrome": [
                "chrome.exe",
            ],
            "google chrome": [
                "chrome.exe",
            ],
            "brave": [
                "brave.exe",
            ],
            "discord": [
                "Discord.exe",
            ],
            "outlook": [
                "OUTLOOK.EXE",
            ],
            "teams": [
                "ms-teams.exe",
                "Teams.exe",
            ],
            "zoom": [
                "Zoom.exe",
            ],
            "obsidian": [
                "Obsidian.exe",
            ],
            "notion": [
                "Notion.exe",
            ],
            "cursor": [
                "Cursor.exe",
            ],
            "code": [
                "Code.exe",
            ],
            "visual studio code": [
                "Code.exe",
            ],
        }

        if target in processes:
            for process in processes[target]:
                self._kill_process(process)

            return f"Closing {target}."

        app = self._find_windows_app(
            [target]
        )

        if app:
            process_name = self._guess_process_name(
                app["name"]
            )

            if process_name:
                self._kill_process(
                    process_name
                )

                return (
                    f"Closing {app['name']}."
                )

        return (
            f"I don't have a safe close action "
            f"configured for {target}."
        )

    # ============================================================
    # PROCESS HELPERS
    # ============================================================

    @staticmethod
    def _kill_process(
        process_name: str,
    ):
        try:
            subprocess.run(
                [
                    "taskkill",
                    "/IM",
                    process_name,
                    "/F",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

        except Exception as exc:
            print(
                f"Process close error: {exc}"
            )

    @staticmethod
    def _guess_process_name(
        app_name: str,
    ):
        mappings = {
            "google chrome": "chrome.exe",
            "brave": "brave.exe",
            "whatsapp": "WhatsApp.exe",
            "discord": "Discord.exe",
            "zoom workplace": "Zoom.exe",
            "cursor": "Cursor.exe",
            "visual studio code": "Code.exe",
            "obsidian": "Obsidian.exe",
            "notion": "Notion.exe",
            "outlook": "OUTLOOK.EXE",
        }

        return mappings.get(
            app_name.lower()
        )

    # ============================================================
    # WEATHER
    # ============================================================

    def _get_weather(self) -> str:
        """
        Lightweight weather lookup.
        """

        try:
            import requests

            response = requests.get(
                "https://wttr.in/?format=3",
                timeout=5,
            )

            if response.ok:
                return response.text.strip()

        except Exception as exc:
            print(
                f"Weather lookup error: {exc}"
            )

        return (
            "I couldn't retrieve the weather right now."
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _contains_any(
        text: str,
        phrases: list[str],
    ) -> bool:
        return any(
            phrase in text
            for phrase in phrases
        )