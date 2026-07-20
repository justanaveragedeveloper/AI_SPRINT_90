"use strict";

const GatewayPriorityQueue = require("../queue/GatewayPriorityQueue");
const GatewayWorkerPool = require("../worker/GatewayWorkerPool");
const ActiveTaskRegistry = require("../worker/ActiveTaskRegistry");

describe("GatewayPriorityQueue", () => {
  let workerPool;
  let activeTaskRegistry;
  let queue;

  beforeEach(() => {
    workerPool = new GatewayWorkerPool(2);

    activeTaskRegistry = new ActiveTaskRegistry();

    queue = new GatewayPriorityQueue({
      workerPool,
      activeTaskRegistry,
    });
  });

  // ========================================================
  // Constructor
  // ========================================================

  describe("constructor()", () => {
    test("creates empty queue", () => {
      expect(queue.size()).toBe(0);
      expect(queue.isEmpty()).toBe(true);
    });

    test("requires worker pool", () => {
      expect(() => {
        new GatewayPriorityQueue({
          activeTaskRegistry,
        });
      }).toThrow("Worker pool is required.");
    });

    test("requires active task registry", () => {
      expect(() => {
        new GatewayPriorityQueue({
          workerPool,
        });
      }).toThrow("Active task registry is required.");
    });
  });

  // ========================================================
  // Enqueue
  // ========================================================

  describe("enqueue()", () => {
    test("rejects invalid task", () => {
      expect(() => {
        queue.enqueue({});
      }).toThrow("Invalid task.");
    });

    test("executes valid task", async () => {
      const execute = jest.fn();

      queue.enqueue({
        id: "task-1",
        weight: 1,
        execute,
      });

      await Promise.resolve();

      expect(execute).toHaveBeenCalledTimes(1);
    });
  });

  // ========================================================
  // Priority
  // ========================================================

  describe("priority ordering", () => {
    test("lower weight executes first", async () => {
      const order = [];

      queue.enqueue({
        id: "high",
        weight: 1,
        execute: jest.fn(() => order.push("high")),
      });

      queue.enqueue({
        id: "low",
        weight: 10,
        execute: jest.fn(() => order.push("low")),
      });

      await Promise.resolve();

      expect(order).toEqual(["high", "low"]);
    });

    test("preserves FIFO for equal priorities", async () => {
      const order = [];

      queue.enqueue({
        id: "one",
        weight: 5,
        execute: jest.fn(() => order.push("one")),
      });

      queue.enqueue({
        id: "two",
        weight: 5,
        execute: jest.fn(() => order.push("two")),
      });

      await Promise.resolve();

      expect(order).toEqual(["one", "two"]);
    });
  });

  // ========================================================
  // Queue Helpers
  // ========================================================

  describe("queue helpers", () => {
    test("peek returns undefined when empty", () => {
      expect(queue.peek()).toBeUndefined();
    });

    test("dequeue returns first task", () => {
      const task = {
        id: "task",
        weight: 1,
        execute: jest.fn(),
      };

      queue.queue.push(task);

      expect(queue.dequeue()).toEqual(task);
      expect(queue.isEmpty()).toBe(true);
    });

    test("removes queued task by id", () => {
      queue.queue.push({
        id: "remove-me",
        weight: 1,
        execute: jest.fn(),
      });

      expect(queue.removeTaskById("remove-me")).toBe(true);
      expect(queue.size()).toBe(0);
    });

    test("returns false for missing task", () => {
      expect(queue.removeTaskById("missing")).toBe(false);
    });
  });

  // ========================================================
  // Worker Lifecycle
  // ========================================================

  describe("worker lifecycle", () => {
    test("releases worker after successful execution", async () => {
      queue.enqueue({
        id: "task",
        weight: 1,
        execute: jest.fn(),
      });

      await Promise.resolve();

      expect(workerPool.activeWorkerCount()).toBe(0);
      expect(workerPool.availableWorkerCount()).toBe(2);
    });

    test("releases worker after failed execution", async () => {
      queue.enqueue({
        id: "task",
        weight: 1,
        execute: jest.fn(() => {
          throw new Error("failure");
        }),
      });

      await Promise.resolve();

      expect(workerPool.activeWorkerCount()).toBe(0);
      expect(workerPool.availableWorkerCount()).toBe(2);
    });
  });

  // ========================================================
  // Active Task Registry
  // ========================================================

  describe("active task registry", () => {
    test("registry is empty after successful execution", async () => {
      queue.enqueue({
        id: "task",
        weight: 1,
        execute: jest.fn(),
      });

      await Promise.resolve();

      expect(activeTaskRegistry.size()).toBe(0);
    });

    test("registry is empty after failed execution", async () => {
      queue.enqueue({
        id: "task",
        weight: 1,
        execute: jest.fn(() => {
          throw new Error("failure");
        }),
      });

      await Promise.resolve();

      expect(activeTaskRegistry.size()).toBe(0);
    });
  });
});
