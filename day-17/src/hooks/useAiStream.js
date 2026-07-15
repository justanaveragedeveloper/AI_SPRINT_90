import { useState, useCallback, useRef, useEffect } from "react";

export const useAiStream = (endpoint) => {
  const [data, setData] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);

  const streamQuery = useCallback(
    async (query, userId) => {
      // Cancel previous stream if one exists
      abortControllerRef.current?.abort();

      const controller = new AbortController();
      abortControllerRef.current = controller;

      setData("");
      setError(null);
      setIsLoading(true);

      // Optional timeout protection
      const timeoutId = setTimeout(() => {
        controller.abort();
      }, 30000); // 30 seconds

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query,
            userId,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Defensive programming
        if (!response.body) {
          throw new Error("Streaming response body unavailable.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          if (value) {
            const chunk = decoder.decode(value, {
              stream: true,
            });

            // Functional update prevents stale state bugs
            setData((prev) => prev + chunk);
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Unexpected streaming error occurred.");
        }
      } finally {
        clearTimeout(timeoutId);

        // Prevent race condition between overlapping requests
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
          setIsLoading(false);
        }
      }
    },
    [endpoint],
  );

  const abortStream = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return {
    data,
    isLoading,
    error,
    streamQuery,
    abortStream,
  };
};