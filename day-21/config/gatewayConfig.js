"use strict";

/**
 * ==========================================================
 * AI Gateway Configuration
 * ==========================================================
 *
 * Centralized configuration used throughout the gateway.
 * Keeping configuration in one place makes the application
 * easier to maintain and avoids magic numbers.
 */

// ==========================================================
// Defaults
// ==========================================================

const DEFAULT_MAX_CONCURRENCY = 4;
const DEFAULT_CACHE_CAPACITY = 500;
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

// ==========================================================
// Environment Variables
// ==========================================================

const MAX_AI_CONCURRENCY = Number.parseInt(
  process.env.MAX_AI_CONCURRENCY,
  10,
);

const PYTHON_AI_SERVICE_URL =
  process.env.PYTHON_AI_SERVICE_URL ??
  "http://localhost:5001/stream";

// ==========================================================
// Safe Configuration
// ==========================================================

const gatewayConfig = {
  /**
   * Maximum number of AI requests that can execute
   * simultaneously.
   */
  maxConcurrency:
    Number.isInteger(MAX_AI_CONCURRENCY) &&
    MAX_AI_CONCURRENCY > 0
      ? MAX_AI_CONCURRENCY
      : DEFAULT_MAX_CONCURRENCY,

  /**
   * Maximum number of cached responses.
   */
  cacheCapacity: DEFAULT_CACHE_CAPACITY,

  /**
   * Timeout for Python inference service.
   */
  requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,

  /**
   * Python AI Streaming Endpoint.
   */
  pythonServiceUrl: PYTHON_AI_SERVICE_URL,
};

// ==========================================================
// Exports
// ==========================================================

module.exports = gatewayConfig;