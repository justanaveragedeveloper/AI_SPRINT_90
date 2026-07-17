// aiWorkerPool.js
import { AiTaskPriorityQueue } from "./aiPriorityQueue19.js";

/**
 * @typedef {Object} AiTask
 * @property {string} taskId - Unique identifier for the task.
 * @property {number} priorityScore - Higher numbers indicate higher priority.
 */

export class AiWorkerPool {
  /**
   * @param {number} maxConcurrency Maximum number of concurrent tasks.
   * @param {{ error: Function }} logger Optional logger.
   */
  constructor(maxConcurrency = 3, logger = console) {
    if (!Number.isInteger(maxConcurrency) || maxConcurrency <= 0) {
      throw new TypeError("maxConcurrency must be a positive integer.");
    }

    this.queue = new AiTaskPriorityQueue();
    this.maxConcurrency = maxConcurrency;
    this.activeCount = 0;
    this.logger = logger;
  }

  /**
   * Submit a task for execution.
   *
   * @param {AiTask} task
   * @param {Function} executionFn Async function representing the workload.
   */
  submitTask(task, executionFn) {
    if (!task || typeof task !== "object") {
      throw new TypeError("Task must be an object.");
    }

    if (typeof task.taskId !== "string" || task.taskId.length === 0) {
      throw new TypeError("taskId must be a non-empty string.");
    }

    if (typeof task.priorityScore !== "number") {
      throw new TypeError("priorityScore must be a number.");
    }

    if (typeof executionFn !== "function") {
      throw new TypeError("executionFn must be a function.");
    }

    this.queue.enqueue({
      taskId: task.taskId,
      priorityScore: task.priorityScore,
      executionFn,
    });

    this.processNext();
  }

  /**
   * Fill all available worker slots.
   */
  processNext() {
    while (this.activeCount < this.maxConcurrency && this.queue.size() > 0) {
      const currentTask = this.queue.dequeue();
      this.activeCount++;

      this.executeTask(currentTask);
    }
  }

  /**
   * Execute one task asynchronously.
   *
   * @private
   */
  async executeTask(task) {
    try {
      await task.executionFn();
    } catch (err) {
      this.logger.error(`Execution error on task ${task.taskId}:`, err);
    } finally {
      this.activeCount--;
      this.processNext();
    }
  }

  getActiveCount() {
    return this.activeCount;
  }
}
