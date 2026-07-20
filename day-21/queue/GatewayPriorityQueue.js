"use strict";

const logger = require("../utils/logger");

/**
 * ==========================================================
 * Gateway Priority Queue
 * ==========================================================
 *
 * Responsibilities:
 *
 * 1. Store queued tasks
 * 2. Sort tasks by priority
 * 3. Preserve FIFO for equal priorities
 * 4. Execute tasks when workers are available
 * 5. Manage worker lifecycle
 *
 * This class knows NOTHING about:
 * - Express
 * - HTTP
 * - Validation
 * - Streaming internals
 * - AI Models
 */

class GatewayPriorityQueue {
  constructor({ workerPool, activeTaskRegistry }) {
    if (!workerPool) {
      throw new Error("Worker pool is required.");
    }

    if (!activeTaskRegistry) {
      throw new Error("Active task registry is required.");
    }

    this.workerPool = workerPool;
    this.activeTaskRegistry = activeTaskRegistry;

    this.queue = [];
    this.sequence = 0;

    this.processing = false;
  }

  // ========================================================
  // Queue Information
  // ========================================================

  size() {
    return this.queue.length;
  }

  isEmpty() {
    return this.queue.length === 0;
  }

  peek() {
    return this.queue[0];
  }

  // ========================================================
  // Enqueue
  // ========================================================

  enqueue(task) {
    if (!task || typeof task.execute !== "function") {
      throw new Error("Invalid task.");
    }

    this.queue.push({
      ...task,
      sequence: this.sequence++,
    });

    this.sortQueue();

    void this.processQueue();
  }

  // ========================================================
  // Queue Sorting
  // ========================================================

  sortQueue() {
    this.queue.sort((a, b) => {
      if (a.weight !== b.weight) {
        return a.weight - b.weight;
      }

      return a.sequence - b.sequence;
    });
  }

  // ========================================================
  // Dequeue
  // ========================================================

  dequeue() {
    return this.queue.shift();
  }

  // ========================================================
  // Remove Waiting Task
  // ========================================================

  removeTaskById(taskId) {
    const index = this.queue.findIndex((task) => task.id === taskId);

    if (index === -1) {
      return false;
    }

    this.queue.splice(index, 1);

    return true;
  }

  // ========================================================
  // Queue Processing
  // ========================================================

  async processQueue() {
    if (this.processing) {
      return;
    }

    this.processing = true;

    try {
      while (!this.isEmpty() && this.workerPool.hasAvailableWorker()) {
        const task = this.dequeue();

        this.workerPool.acquireWorkerSlot();
        this.activeTaskRegistry.register(task);

        try {
          await task.execute();
        } catch (error) {
          // Temporary logging.
          // Design Problem 5 will replace this with
          // a structured logger.

          logger.error("Task execution failed", {
            taskId: task.id,
            error: error.message,
            stack: error.stack,
          });
        } finally {
          this.activeTaskRegistry.remove(task.id);

          this.workerPool.releaseWorkerSlot();
        }
      }
    } finally {
      this.processing = false;

      if (!this.isEmpty() && this.workerPool.hasAvailableWorker()) {
        void this.processQueue();
      }
    }
  }
}

module.exports = GatewayPriorityQueue;
