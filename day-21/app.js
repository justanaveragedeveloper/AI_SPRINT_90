const express = require("express");

const gatewayConfig = require("./config/gatewayConfig");

const GatewayWorkerPool = require("./worker/GatewayWorkerPool");
const ActiveTaskRegistry = require("./worker/ActiveTaskRegistry");
const GatewayPriorityQueue = require("./queue/GatewayPriorityQueue");

const createAiRoutes = require("./routes/aiRoutes");
const logger = require("./utils/logger");

/**
 * ==========================================================
 * Application Bootstrap
 * ==========================================================
 */

const app = express();

app.use(express.json());

/**
 * ==========================================================
 * Shared Infrastructure
 * ==========================================================
 */

const gatewayPool = new GatewayWorkerPool(gatewayConfig.maxConcurrency);

const activeTaskRegistry = new ActiveTaskRegistry();

/**
 * Temporary cache.
 * Replace with the LRU cache later.
 */

const cache = {
  store: new Map(),

  get(key) {
    return this.store.get(key);
  },

  put(key, value) {
    this.store.set(key, value);
  },
};

const gatewayQueue = new GatewayPriorityQueue({
  workerPool: gatewayPool,
  activeTaskRegistry,
});

/**
 * ==========================================================
 * Routes
 * ==========================================================
 */

app.use(
  "/api/ai",
  createAiRoutes({
    gatewayQueue,
    cache,
    activeTaskRegistry,
  }),
);

/**
 * ==========================================================
 * Health Check
 * ==========================================================
 */

app.get("/health", (_, res) => {
  res.json({
    status: "ok",
  });
});

/**
 * ==========================================================
 * Global Error Handler
 * ==========================================================
 */

app.use((err, req, res, next) => {
  logger.error("Unhandled application error", {
    error: err.message,
    stack: err.stack,
  });

  const statusCode = err.statusCode ?? 500;

  res.status(statusCode).json({
    error: err.message,
  });
});

module.exports = app;
