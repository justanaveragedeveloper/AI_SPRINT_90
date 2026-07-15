// controllers/aiController.js

import { TransactionLog } from "./TransactionLog.js";

const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL ??
  "http://localhost:8000/api/v1/predict/stream";

const REQUEST_TIMEOUT_MS = 30_000;

export const handleAiQuery = async (req, res, next) => {
  const { query, userId } = req.body;

  if (
    typeof query !== "string" ||
    query.trim().length === 0 ||
    typeof userId !== "string" ||
    userId.trim().length === 0
  ) {
    return res.status(400).json({
      success: false,
      error: "query and userId must be non-empty strings.",
    });
  }

  const controller = new AbortController();

  req.on("aborted", () => {
    controller.abort();
  });

  let keepAliveInterval;
  let timeout;

  try {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });

    keepAliveInterval = setInterval(() => {
      if (!res.writableEnded) {
        res.write(": ping\n\n");
      }
    }, 15_000);

    timeout = setTimeout(() => {
      controller.abort();
    }, REQUEST_TIMEOUT_MS);

    const response = await fetch(PYTHON_SERVICE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Python service returned ${response.status}`);
    }

    if (!response.body) {
      throw new Error("Python service returned an empty stream.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      const chunk = decoder.decode(value, {
        stream: true,
      });

      res.write(`data: ${chunk}\n\n`);
    }

    try {
      await TransactionLog.create({
        userId,
        queryLength: query.length,
        timestamp: new Date(),
        status: "COMPLETED",
      });
    } catch (dbError) {
      console.error("Telemetry logging failed:", dbError);
    }

    res.end();
  } catch (error) {
    console.error("Streaming gateway error:", error);

    const message =
      error.name === "AbortError"
        ? "Inference request timed out or client disconnected."
        : error.message;

    if (!res.headersSent) {
      return res.status(500).json({
        success: false,
        error: message,
      });
    }

    if (!res.writableEnded) {
      res.write(`data: [ERROR] ${message}\n\n`);
      res.end();
    }

    if (!res.headersSent) {
      return next(error);
    }
  } finally {
    clearInterval(keepAliveInterval);
    clearTimeout(timeout);
  }
};
