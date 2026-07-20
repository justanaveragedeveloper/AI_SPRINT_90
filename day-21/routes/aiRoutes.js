"use strict";

const express = require("express");

const createAiController = require("../controllers/aiController");

function createAiRoutes({ gatewayQueue, cache, activeTaskRegistry }) {
  const router = express.Router();

  const aiController = createAiController({
    gatewayQueue,
    cache,
    activeTaskRegistry,
  });

  router.post("/generate", aiController);

  return router;
}

module.exports = createAiRoutes;
