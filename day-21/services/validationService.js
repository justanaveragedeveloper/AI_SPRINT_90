"use strict";

const ValidationError = require("../errors/ValidationError");

/**
 * ==========================================================
 * Validation Service
 * ==========================================================
 *
 * Responsible for validating every incoming AI request
 * before it enters the gateway pipeline.
 *
 * This service does NOT send HTTP responses.
 * It only validates data and throws Errors when something
 * is invalid.
 */

// ==========================================================
// Constants
// ==========================================================

const MAX_PROMPT_LENGTH = 50_000;

const ALLOWED_TIERS = new Set(["free", "premium"]);

// ==========================================================
// Prompt Validation
// ==========================================================

function validatePrompt(prompt) {
  if (typeof prompt !== "string") {
    throw new ValidationError("Prompt must be a string.");
  }

  const trimmedPrompt = prompt.trim();

  if (trimmedPrompt.length === 0) {
    throw new ValidationError("Prompt cannot be empty.");
  }

  if (trimmedPrompt.length > MAX_PROMPT_LENGTH) {
    throw new ValidationError("Prompt exceeds maximum allowed size.");
  }

  return trimmedPrompt;
}

// ==========================================================
// Tier Validation
// ==========================================================

function validateTier(tier = "free") {
  const normalizedTier = String(tier).toLowerCase();

  if (!ALLOWED_TIERS.has(normalizedTier)) {
    throw new ValidationError(`Unsupported tier "${tier}".`);
  }

  return normalizedTier;
}

// ==========================================================
// RAG Parameter Validation
// ==========================================================

function validateRagParams(ragParams = {}) {
  if (
    ragParams === null ||
    typeof ragParams !== "object" ||
    Array.isArray(ragParams)
  ) {
    throw new ValidationError("ragParams must be an object.");
  }

  return ragParams;
}

// ==========================================================
// Complete Request Validation
// ==========================================================

/**
 * Validates an entire AI request.
 *
 * Returns a cleaned object that can safely be
 * passed to downstream services.
 */
function validateRequest(body = {}) {
  const { prompt, tier = "free", ragParams = {} } = body;

  return {
    prompt: validatePrompt(prompt),
    tier: validateTier(tier),
    ragParams: validateRagParams(ragParams),
  };
}

// ==========================================================
// Exports
// ==========================================================

module.exports = {
  validatePrompt,
  validateTier,
  validateRagParams,
  validateRequest,
};
