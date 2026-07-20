"use strict";

const { validateRequest } = require("../services/validationService");

const {
  buildCacheKey,
  tryServeCachedResponse,
} = require("../services/cacheService");

const { createGatewayTask } = require("../services/taskFactory");

const { cleanupAbortedTask } = require("../utils/workerHelpers");

/**
 * ==========================================================
 * AI Controller
 * ==========================================================
 *
 * Responsibilities:
 *
 * 1. Validate request
 * 2. Check cache
 * 3. Create task
 * 4. Register disconnect handler
 * 5. Enqueue task
 *
 * The controller contains NO business logic.
 */

function createAiController({ gatewayQueue, cache, activeTaskRegistry }) {
  return async function aiController(req, res, next) {
    try {
      const { prompt, tier, ragParams } = validateRequest(req.body);

      const cacheKey = buildCacheKey(prompt, ragParams);

      const cacheHit = tryServeCachedResponse({
        cache,
        cacheKey,
        res,
      });

      if (cacheHit) {
        return;
      }

      const abortController = new AbortController();

      const task = createGatewayTask({
        prompt,
        tier,
        ragParams,
        cache,
        cacheKey,
        res,
        abortController,
      });

      req.on("close", () => {
        cleanupAbortedTask({
          gatewayQueue,
          activeTaskRegistry,
          task,
          abortController,
        });
      });

      gatewayQueue.enqueue(task);
    } catch (error) {
      next(error);
    }
  };
}

module.exports = createAiController;
