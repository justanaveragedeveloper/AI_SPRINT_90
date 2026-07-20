"use strict";

/**
 * ==========================================================
 * Active Task Registry
 * ==========================================================
 *
 * Responsible ONLY for tracking tasks that are currently
 * executing inside the gateway.
 *
 * Responsibilities:
 * 1. Register active tasks
 * 2. Remove completed tasks
 * 3. Lookup active tasks
 *
 * This class knows NOTHING about:
 * - Worker capacity
 * - Queues
 * - Express
 * - Streaming
 */

class ActiveTaskRegistry {
  constructor() {
    this.tasks = new Map();
  }

  register(task) {
    if (!task || !task.id) {
      throw new Error("Task must have an id.");
    }

    this.tasks.set(task.id, task);
  }

  remove(taskId) {
    return this.tasks.delete(taskId);
  }

  get(taskId) {
    return this.tasks.get(taskId);
  }

  has(taskId) {
    return this.tasks.has(taskId);
  }

  size() {
    return this.tasks.size;
  }

  clear() {
    this.tasks.clear();
  }
}

module.exports = ActiveTaskRegistry;
