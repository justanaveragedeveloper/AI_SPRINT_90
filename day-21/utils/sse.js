"use strict";

/**
 * ==========================================================
 * Server-Sent Events (SSE) Utilities
 * ==========================================================
 *
 * This file contains helper functions for sending
 * Server-Sent Events (SSE) responses to the client.
 *
 * By centralizing SSE logic here, controllers and services
 * don't need to worry about HTTP header details or
 * formatting SSE messages.
 */

// ==========================================================
// SSE Response Headers
// ==========================================================

/**
 * Sends the required HTTP headers for an SSE connection.
 *
 * @param {import("express").Response} res
 * @param {"HIT"|"MISS"} cacheStatus
 */
function writeSseHeaders(res, cacheStatus) {
  if (res.headersSent) {
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Cache-Status": cacheStatus,
  });
}

// ==========================================================
// SSE Data
// ==========================================================

/**
 * Sends one SSE event to the client.
 *
 * Example output:
 *
 * data: {"text":"Hello","done":false}
 *
 *
 * @param {import("express").Response} res
 * @param {Object} payload
 */
function writeSseData(res, payload) {
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

// ==========================================================
// SSE Completion
// ==========================================================

/**
 * Sends the final completion event and closes
 * the HTTP connection.
 *
 * @param {import("express").Response} res
 */
function finishSseStream(res) {
  writeSseData(res, {
    done: true,
  });

  res.end();
}

// ==========================================================
// SSE Error
// ==========================================================

/**
 * Sends an SSE error event.
 *
 * Used when headers have already been sent and
 * we cannot switch to a normal JSON response.
 *
 * @param {import("express").Response} res
 * @param {string} message
 */
function writeSseError(res, message) {
  writeSseData(res, {
    error: message,
  });

  res.end();
}

// ==========================================================
// Exports
// ==========================================================

module.exports = {
  writeSseHeaders,
  writeSseData,
  finishSseStream,
  writeSseError,
};
