"use strict";

const crypto = require("crypto");

const estimateTokenLength = require("../utils/tokenEstimator");

const { streamPythonResponse } = require("./streamService");

/**
 * ==========================================================
 * Gateway Task Factory
 * ==========================================================
 *
 * Responsible ONLY for creating executable gateway tasks.
 */

function calculateTaskWeight(prompt, tier) {
  const tokenCount = estimateTokenLength(prompt);

  const tierModifier = tier === "premium" ? 1 : 5;

  return tierModifier * tokenCount;
}

function createGatewayTask({
  prompt,
  tier,
  ragParams,

  cache,
  cacheKey,

  res,

  abortController,
}) {
  return {
    id: crypto.randomUUID(),

    weight: calculateTaskWeight(prompt, tier),

    prompt,

    ragParams,

    abortController,

    async execute() {
      if (abortController.signal.aborted) {
        return;
      }

      await streamPythonResponse({
        res,
        prompt,
        ragParams,
        cache,
        cacheKey,
        abortController,
      });
    },
  };
}

module.exports = {
  calculateTaskWeight,
  createGatewayTask,
};
