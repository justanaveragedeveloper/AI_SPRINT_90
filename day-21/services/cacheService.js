"use strict";

const generateCacheKey = require("../utils/cacheKey");
const {
  writeSseHeaders,
  writeSseData,
  finishSseStream,
} = require("../utils/sse");

/**
 * ==========================================================
 * Cache Service
 * ==========================================================
 *
 * Responsible for all cache-related operations.
 *
 * The controller should never directly access the LRU cache.
 * Instead, it asks this service to:
 *
 * 1. Generate cache keys
 * 2. Read cached responses
 * 3. Store completed responses
 * 4. Serve cached SSE responses
 */

// ==========================================================
// Cache Key
// ==========================================================

function buildCacheKey(prompt, ragParams) {
  return generateCacheKey(prompt, ragParams);
}

// ==========================================================
// Cache Read
// ==========================================================

function getCachedResponse(cache, cacheKey) {
  return cache.get(cacheKey);
}

// ==========================================================
// Cache Write
// ==========================================================

function storeResponse(cache, cacheKey, response) {
  cache.put(cacheKey, response);
}

// ==========================================================
// Cache Hit Response
// ==========================================================

/**
 * Sends a cached response to the client using
 * Server-Sent Events.
 *
 * Returns:
 * true  -> cache hit
 * false -> cache miss
 */
function tryServeCachedResponse({ cache, cacheKey, res }) {
  const cachedResponse = getCachedResponse(cache, cacheKey);

  if (!cachedResponse) {
    return false;
  }

  writeSseHeaders(res, "HIT");

  writeSseData(res, {
    text: cachedResponse,
    done: true,
  });

  finishSseStream(res);

  return true;
}

// ==========================================================
// Exports
// ==========================================================

module.exports = {
  buildCacheKey,
  getCachedResponse,
  storeResponse,
  tryServeCachedResponse,
};
