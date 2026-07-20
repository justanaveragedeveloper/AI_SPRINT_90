"use strict";

/**
 * ==========================================================
 * Simple Structured Logger
 * ==========================================================
 *
 * Provides consistent application logging.
 *
 * This implementation intentionally avoids external
 * dependencies. It can later be replaced by Winston,
 * Pino, or another logging library without changing
 * the rest of the application.
 */

function log(level, message, metadata = {}) {
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    ...metadata,
  };

  const serialized = JSON.stringify(entry);

  switch (level) {
    case "ERROR":
      console.error(serialized);
      break;

    case "WARN":
      console.warn(serialized);
      break;

    default:
      console.log(serialized);
  }
}

module.exports = {
  info(message, metadata) {
    log("INFO", message, metadata);
  },

  warn(message, metadata) {
    log("WARN", message, metadata);
  },

  error(message, metadata) {
    log("ERROR", message, metadata);
  },
};
