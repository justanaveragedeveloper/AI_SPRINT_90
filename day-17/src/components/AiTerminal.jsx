import React, { useState } from "react";
import { useAiStream } from "../hooks/useAiStream";

const MAX_PROMPT_LENGTH = 5000;

const AiTerminal = () => {
  const [inputQuery, setInputQuery] = useState("");

  const { data, isLoading, error, streamQuery, abortStream } = useAiStream(
    "http://localhost:3000/api/v1/ai/query",
  );

  const handleSubmit = async (e) => {
    e.preventDefault();

    const trimmedQuery = inputQuery.trim();

    if (!trimmedQuery) {
      return;
    }

    if (trimmedQuery.length > MAX_PROMPT_LENGTH) {
      return;
    }

    try {
      // Replace with authenticated user id in production
      await streamQuery(trimmedQuery, "user_dev_90");

      // Clear textarea after successful submission
      setInputQuery("");
    } catch (err) {
      console.error("Failed to initiate stream:", err);
    }
  };

  const remainingCharacters = MAX_PROMPT_LENGTH - inputQuery.length;

  return (
    <div
      style={{
        padding: "24px",
        fontFamily: "monospace",
        maxWidth: "800px",
        margin: "0 auto",
      }}
    >
      <h2>System 17: Token Streaming Monitor</h2>

      <form
        onSubmit={handleSubmit}
        style={{
          marginBottom: "16px",
        }}
      >
        <textarea
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          placeholder="Enter analytical model query..."
          rows={4}
          maxLength={MAX_PROMPT_LENGTH}
          disabled={isLoading}
          style={{
            width: "100%",
            padding: "12px",
            background: "#1e1e1e",
            color: "#ffffff",
            borderRadius: "4px",
            resize: "vertical",
          }}
        />

        <div
          style={{
            marginTop: "8px",
            fontSize: "12px",
            opacity: 0.7,
          }}
        >
          Characters remaining: {remainingCharacters}
        </div>

        <div
          style={{
            marginTop: "10px",
            display: "flex",
            gap: "10px",
          }}
        >
          <button
            type="submit"
            disabled={isLoading || !inputQuery.trim()}
            style={{
              padding: "8px 16px",
              cursor: isLoading ? "not-allowed" : "pointer",
            }}
          >
            {isLoading ? "Streaming Tokens..." : "Execute Query"}
          </button>

          {isLoading && (
            <button
              type="button"
              onClick={abortStream}
              style={{
                padding: "8px 16px",
                background: "#d9534f",
                color: "#ffffff",
                border: "none",
                cursor: "pointer",
              }}
            >
              Stop Generation
            </button>
          )}
        </div>
      </form>

      {error && (
        <div
          style={{
            color: "#ff6b6b",
            marginBottom: "16px",
          }}
        >
          ⚠️ Streaming Error: {error}
        </div>
      )}

      <div
        style={{
          background: "#0f0f0f",
          color: "#39ff14",
          padding: "16px",
          borderRadius: "4px",
          minHeight: "200px",
          whiteSpace: "pre-wrap",
          overflowWrap: "break-word",
          overflowX: "auto",
        }}
      >
        {data ||
          (isLoading
            ? "🔄 Awaiting first token..."
            : "⚡ System idle. Awaiting prompt submission.")}
      </div>
    </div>
  );
};

export default AiTerminal;