"use strict";

const crypto = require("crypto");
const stableStringify = require("./stableStringify");

/**
 * ==========================================================
 * Cache Key Generator
 * ==========================================================
 *
 * Generates a deterministic SHA-256 hash that uniquely
 * identifies an AI request.
 *
 * The cache key depends on:
 *   1. User prompt
 *   2. RAG parameters
 *
 * If either changes, a completely new cache key is produced.
 */

/**
 * Creates a deterministic cache key.
 *
 * @param {string} prompt
 * @param {Object} ragParams
 * @returns {string}
 */
function generateCacheKey(prompt, ragParams = {}) {
  const hash = crypto.createHash("sha256");

  hash.update(prompt);
  hash.update(":");
  hash.update(stableStringify(ragParams));

  return hash.digest("hex");
}

module.exports = generateCacheKey;
