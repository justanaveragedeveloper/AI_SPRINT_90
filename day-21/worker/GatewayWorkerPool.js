"use strict";

/**
 * ==========================================================
 * Gateway Worker Pool
 * ==========================================================
 *
 * Responsible ONLY for limiting concurrent execution.
 *
 * Responsibilities:
 * 1. Acquire worker slots
 * 2. Release worker slots
 * 3. Report worker availability
 *
 * This class intentionally does NOT know:
 * - Tasks
 * - Queues
 * - Express
 * - Streaming
 * - Active task registry
 */

class GatewayWorkerPool {
  constructor(maxWorkers = 4) {
    if (!Number.isInteger(maxWorkers) || maxWorkers <= 0) {
      throw new Error("maxWorkers must be a positive integer.");
    }

    this.maxWorkers = maxWorkers;
    this.activeWorkers = 0;
  }

  acquireWorkerSlot() {
    if (!this.hasAvailableWorker()) {
      throw new Error("No worker slots available.");
    }

    this.activeWorkers++;
  }

  releaseWorkerSlot() {
    if (this.activeWorkers > 0) {
      this.activeWorkers--;
    }
  }

  hasAvailableWorker() {
    return this.activeWorkers < this.maxWorkers;
  }

  activeWorkerCount() {
    return this.activeWorkers;
  }

  availableWorkerCount() {
    return this.maxWorkers - this.activeWorkers;
  }

  clear() {
    this.activeWorkers = 0;
  }
}

module.exports = GatewayWorkerPool;
