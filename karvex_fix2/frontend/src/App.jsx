import React, { useEffect, useRef, useState } from "react";
import "./styles.css";

const API = "https://karvex-ai-assistant.onrender.com";
const IDLE = "idle";
const LISTENING = "listening";
const THINKING = "thinking";
const SPEAKING = "speaking";

function normalizeAssistantName(text) {
  const legacyName = String.fromCharCode(114, 97, 118, 105, 101, 108);
  return String(text ?? "").replace(new RegExp(`\\b${legacyName}\\b`, "gi"), "KARVEX");
}

// ============================================================
// VOICE
// ============================================================

function useVoice({ onQuestion, onResponse, onError } = {}) {
  const recognitionRef =
    useRef(null);

  const requestRef =
    useRef(null);

  const [state, setState] =
    useState(IDLE);


  useEffect(() => {
    return () => {
      try {
        recognitionRef.current?.abort();
      } catch {}

      try {
        requestRef.current?.abort();
      } catch {}

      try {
        window.speechSynthesis.cancel();
      } catch {}
    };
  }, []);


  function speak(text) {
    if (!text?.trim()) {
      setState(IDLE);
      return;
    }

    window.speechSynthesis.cancel();

    const utterance =
      new SpeechSynthesisUtterance(
        text.trim()
      );

    utterance.lang =
      "en-US";

    utterance.rate =
      1.25;

    utterance.pitch =
      0.96;

    utterance.volume = 1;


    utterance.onstart =
      () => {
        setState(SPEAKING);
      };


    utterance.onend =
      () => {
        setState(IDLE);
      };


    utterance.onerror =
      () => {
        setState(IDLE);
      };


    window.speechSynthesis.speak(
      utterance
    );
  }


  async function ask(
    query
  ) {
    const controller = new AbortController();
    requestRef.current = controller;

    // Prevent the UI from getting stuck forever if the API/network hangs.
    const timeoutId = window.setTimeout(() => controller.abort(), 60000);

    setState(THINKING);
    onQuestion?.(query);

    try {
      const response = await fetch(`${API}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Support the current API shape and common wrapped response shapes.
      const answer = [
        data?.answer,
        data?.data?.answer,
        data?.response,
        data?.message,
      ].find((value) => typeof value === "string" && value.trim());

      if (!answer) {
        throw new Error("The AI returned an empty answer.");
      }

      // Update the response panel and leave THINKING before speech starts.
      // This guarantees the answer is visible even if browser TTS is delayed.
      setState(IDLE);
      const cleanAnswer = normalizeAssistantName(answer.trim());
      onResponse?.({ query, answer: cleanAnswer });
      speak(cleanAnswer);
    } catch (error) {
      if (error.name === "AbortError") {
        const message =
          "KARVEX could not complete the request. Please try again.";
        setState(IDLE);
        onError?.(message);
        return;
      }

      console.error("[KARVEX]", error);
      setState(IDLE);
      onError?.(
        error?.message ||
          "I could not connect to the intelligence system."
      );
    } finally {
      window.clearTimeout(timeoutId);
      requestRef.current = null;
    }
  }

  function startListening() {
    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;


    if (!SpeechRecognition) {
      console.error(
        "Speech recognition is not supported."
      );

      return;
    }


    try {
      recognitionRef.current?.abort();
    } catch {}


    window.speechSynthesis.cancel();


    const recognition =
      new SpeechRecognition();


    // ENGLISH ONLY
    recognition.lang =
      "en-US";

    recognition.continuous =
      false;

    recognition.interimResults =
      false;

    recognition.maxAlternatives =
      1;


    recognition.onstart =
      () => {
        setState(LISTENING);
      };


    recognition.onresult =
      event => {
        const transcript =
          event
            ?.results?.[0]?.[0]
            ?.transcript
            ?.trim();


        if (!transcript) {
          setState(IDLE);
          return;
        }


        try {
          recognition.stop();
        } catch {}


        ask(transcript);
      };


    recognition.onerror =
      event => {
        console.error(
          "[KARVEX] microphone:",
          event.error
        );

        setState(IDLE);
      };


    recognition.onend =
      () => {
        setState(
          current =>
            current === LISTENING
              ? IDLE
              : current
        );
      };


    recognitionRef.current =
      recognition;


    try {
      recognition.start();
    } catch (error) {
      console.error(
        "[KARVEX]",
        error
      );

      setState(IDLE);
    }
  }


  function toggle() {
    if (
      state === THINKING ||
      state === SPEAKING
    ) {
      return;
    }


    if (
      state === LISTENING
    ) {
      try {
        recognitionRef.current?.stop();
      } catch {}

      setState(IDLE);

      return;
    }


    startListening();
  }


  return {
    state,
    toggle,
    ask,
  };
}




// ============================================================
// KARVEX UI
// ============================================================

function KarvexVisual({ state, onMic, onAsk, question, answer, error }) {
  const [query, setQuery] = useState("");

  function submit(event) {
    event?.preventDefault();
    const value = query.trim();
    if (!value || state === THINKING || state === SPEAKING) return;
    onAsk(value);
    setQuery("");
  }

  const hasResponse = Boolean(question || answer || error);
  const summary = answer
    ? answer.replace(/\s+/g, " ").trim().slice(0, 360) +
      (answer.replace(/\s+/g, " ").trim().length > 360 ? "…" : "")
    : "";

  return (
    <main className="karvex-app">
      <img
        className="karvex-background"
        src="/karvex-dashboard.png"
        alt="KARVEX futuristic AI assistant interface"
        draggable="false"
      />

      <section className={`karvex-response-panel ${hasResponse ? "has-response" : ""}`} aria-live="polite">
        <div className="response-title">
          <span className="response-dot" aria-hidden="true" />
          KARVEX · RESPONSE
        </div>

        {state === THINKING && !answer && !error ? (
          <div className="response-thinking">KARVEX IS THINKING...</div>
        ) : error ? (
          <div className="response-error">{error}</div>
        ) : hasResponse ? (
          <>
            <div className="response-label">YOUR QUESTION</div>
            <div className="response-question">{question}</div>
            <div className="response-divider" />
            <div className="response-label">WHAT KARVEX ANSWERED</div>
            <div className="response-answer">{summary}</div>
          </>
        ) : (
          <div className="response-empty">
            Ask KARVEX anything and your latest answer will appear here.
          </div>
        )}
      </section>

      <div className="karvex-search-shell">
        <form className="karvex-search-form" onSubmit={submit}>
          <input
            className="karvex-search-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type your message here..."
            aria-label="Ask KARVEX"
            autoComplete="off"
            spellCheck="false"
          />
          <button
            className="karvex-send-button"
            type="submit"
            aria-label="Send message"
            disabled={!query.trim() || state === THINKING || state === SPEAKING}
          >
            <span aria-hidden="true">➤</span>
          </button>
        </form>
      </div>

      <button
        className={`karvex-mic-button ${state}`}
        type="button"
        onClick={onMic}
        aria-label={state === LISTENING ? "Stop listening" : "Speak to KARVEX"}
        disabled={state === THINKING || state === SPEAKING}
      >
        <span aria-hidden="true">🎙</span>
      </button>

      <div className="karvex-status" aria-live="polite">
        {state === THINKING && "KARVEX IS THINKING..."}
        {state === LISTENING && "LISTENING..."}
        {state === SPEAKING && "KARVEX IS SPEAKING..."}
      </div>
    </main>
  );
}

// ============================================================
// APP
// ============================================================

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");

  const { state, toggle, ask } = useVoice({
    onQuestion: (query) => {
      setQuestion(query);
      setAnswer("");
      setError("");
    },
    onResponse: ({ query, answer: nextAnswer }) => {
      setQuestion(query);
      setAnswer(nextAnswer);
      setError("");
    },
    onError: (message) => {
      setError(message);
    },
  });

  function handleAsk(query) {
    setQuestion(query);
    setAnswer("");
    setError("");
    ask(query);
  }

  return (
    <KarvexVisual
      state={state}
      onMic={toggle}
      onAsk={handleAsk}
      question={question}
      answer={answer}
      error={error}
    />
  );
}
