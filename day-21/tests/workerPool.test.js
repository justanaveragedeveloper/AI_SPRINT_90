"use strict";

const GatewayWorkerPool = require("../worker/GatewayWorkerPool");

describe("GatewayWorkerPool", () => {
  let workerPool;

  beforeEach(() => {
    workerPool = new GatewayWorkerPool(2);
  });

  // ========================================================
  // Constructor
  // ========================================================

  describe("constructor()", () => {
    test("creates worker pool", () => {
      expect(workerPool.maxWorkers).toBe(2);
      expect(workerPool.activeWorkerCount()).toBe(0);
      expect(workerPool.availableWorkerCount()).toBe(2);
    });

    test("rejects zero workers", () => {
      expect(() => {
        new GatewayWorkerPool(0);
      }).toThrow("maxWorkers must be a positive integer.");
    });

    test("rejects negative workers", () => {
      expect(() => {
        new GatewayWorkerPool(-1);
      }).toThrow("maxWorkers must be a positive integer.");
    });

    test("rejects non-integer workers", () => {
      expect(() => {
        new GatewayWorkerPool(1.5);
      }).toThrow("maxWorkers must be a positive integer.");
    });
  });

  // ========================================================
  // Worker Slot Acquisition
  // ========================================================

  describe("acquireWorkerSlot()", () => {
    test("acquires one worker", () => {
      workerPool.acquireWorkerSlot();

      expect(workerPool.activeWorkerCount()).toBe(1);
      expect(workerPool.availableWorkerCount()).toBe(1);
    });

    test("throws when pool is full", () => {
      workerPool.acquireWorkerSlot();
      workerPool.acquireWorkerSlot();

      expect(() => {
        workerPool.acquireWorkerSlot();
      }).toThrow("No worker slots available.");
    });
  });

  // ========================================================
  // Worker Slot Release
  // ========================================================

  describe("releaseWorkerSlot()", () => {
    test("releases worker", () => {
      workerPool.acquireWorkerSlot();
      workerPool.releaseWorkerSlot();

      expect(workerPool.activeWorkerCount()).toBe(0);
      expect(workerPool.availableWorkerCount()).toBe(2);
    });

    test("never releases below zero", () => {
      workerPool.releaseWorkerSlot();

      expect(workerPool.activeWorkerCount()).toBe(0);
    });
  });

  // ========================================================
  // Availability
  // ========================================================

  describe("availability", () => {
    test("reports available workers", () => {
      expect(workerPool.hasAvailableWorker()).toBe(true);

      workerPool.acquireWorkerSlot();
      workerPool.acquireWorkerSlot();

      expect(workerPool.hasAvailableWorker()).toBe(false);
    });

    test("reports active worker count", () => {
      workerPool.acquireWorkerSlot();

      expect(workerPool.activeWorkerCount()).toBe(1);
    });

    test("reports available worker count", () => {
      workerPool.acquireWorkerSlot();

      expect(workerPool.availableWorkerCount()).toBe(1);
    });
  });

  // ========================================================
  // Reset
  // ========================================================

  describe("clear()", () => {
    test("clears worker state", () => {
      workerPool.acquireWorkerSlot();
      workerPool.acquireWorkerSlot();

      workerPool.clear();

      expect(workerPool.activeWorkerCount()).toBe(0);
      expect(workerPool.availableWorkerCount()).toBe(2);
      expect(workerPool.hasAvailableWorker()).toBe(true);
    });
  });
});
