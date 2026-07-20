"use strict";

const axios = require("axios");

const gatewayConfig = require("../config/gatewayConfig");

const {
  writeSseHeaders,
  writeSseData,
  finishSseStream,
  writeSseError,
} = require("../utils/sse");

const { storeResponse } = require("./cacheService");

/**
 * ==========================================================
 * Stream Service
 * ==========================================================
 *
 * Responsibilities:
 *
 * 1. Call the Python AI service
 * 2. Stream chunks to the client
 * 3. Build the complete response
 * 4. Store completed response in cache
 *
 * This service intentionally knows NOTHING about:
 * - Worker pools
 * - Priority queues
 * - Controllers
 * - Express routing
 */

async function streamPythonResponse({
  res,
  prompt,
  ragParams,
  cache,
  cacheKey,
  abortController,
}) {
  writeSseHeaders(res, "MISS");

  let completeResponse = "";

  try {
    const response = await axios({
      method: "post",
      url: gatewayConfig.pythonServiceUrl,
      timeout: gatewayConfig.requestTimeoutMs,
      signal: abortController.signal,
      responseType: "stream",
      data: {
        prompt,
        ...ragParams,
      },
    });

    await new Promise((resolve, reject) => {
      response.data.on("data", (chunk) => {
        if (abortController.signal.aborted) {
          return;
        }

        const text = chunk.toString();

        completeResponse += text;

        writeSseData(res, {
          text,
          done: false,
        });
      });

      response.data.on("end", () => {
        if (!abortController.signal.aborted) {
          storeResponse(cache, cacheKey, completeResponse);

          finishSseStream(res);
        }

        resolve();
      });

      response.data.on("error", reject);
    });
  } catch (error) {
    if (abortController.signal.aborted || error.name === "CanceledError") {
      return;
    }

    if (!res.headersSent) {
      return res.status(500).json({
        error: "Streaming gateway processing failure.",
      });
    }

    writeSseError(res, "Streaming gateway processing failure.");

    throw error;
  }
}

module.exports = {
  streamPythonResponse,
};
