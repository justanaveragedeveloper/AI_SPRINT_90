"use strict";

const createAiController = require("../controllers/aiController");

const validationService = require("../services/validationService");
const cacheService = require("../services/cacheService");
const taskFactory = require("../services/taskFactory");
const workerHelpers = require("../utils/workerHelpers");

jest.mock("../services/validationService");
jest.mock("../services/cacheService");
jest.mock("../services/taskFactory");
jest.mock("../utils/workerHelpers");

describe("AI Controller", () => {
  let gatewayQueue;
  let cache;
  let activeTaskRegistry;

  let controller;

  let req;
  let res;
  let next;

  beforeEach(() => {
    gatewayQueue = {
      enqueue: jest.fn(),
    };

    cache = {};

    activeTaskRegistry = {};

    controller = createAiController({
      gatewayQueue,
      cache,
      activeTaskRegistry,
    });

    req = {
      body: {},
      on: jest.fn(),
    };

    res = {};

    next = jest.fn();

    jest.clearAllMocks();
  });

  // ========================================================
  // Successful Request
  // ========================================================

  describe("successful request", () => {
    test("creates and enqueues task", async () => {
      validationService.validateRequest.mockReturnValue({
        prompt: "Hello",
        tier: "free",
        ragParams: {},
      });

      cacheService.buildCacheKey.mockReturnValue("cache-key");

      cacheService.tryServeCachedResponse.mockReturnValue(false);

      const task = {
        id: "task-1",
      };

      taskFactory.createGatewayTask.mockReturnValue(task);

      await controller(req, res, next);

      expect(taskFactory.createGatewayTask).toHaveBeenCalled();

      expect(gatewayQueue.enqueue).toHaveBeenCalledWith(task);

      expect(next).not.toHaveBeenCalled();
    });

    test("registers disconnect handler", async () => {
      validationService.validateRequest.mockReturnValue({
        prompt: "Hello",
        tier: "free",
        ragParams: {},
      });

      cacheService.buildCacheKey.mockReturnValue("key");

      cacheService.tryServeCachedResponse.mockReturnValue(false);

      taskFactory.createGatewayTask.mockReturnValue({
        id: "task",
      });

      await controller(req, res, next);

      expect(req.on).toHaveBeenCalledWith("close", expect.any(Function));
    });
  });

  // ========================================================
  // Cache
  // ========================================================

  describe("cache hit", () => {
    test("does not enqueue task", async () => {
      validationService.validateRequest.mockReturnValue({
        prompt: "Hello",
        tier: "free",
        ragParams: {},
      });

      cacheService.buildCacheKey.mockReturnValue("key");

      cacheService.tryServeCachedResponse.mockReturnValue(true);

      await controller(req, res, next);

      expect(taskFactory.createGatewayTask).not.toHaveBeenCalled();

      expect(gatewayQueue.enqueue).not.toHaveBeenCalled();
    });
  });

  // ========================================================
  // Disconnect Handler
  // ========================================================

  describe("disconnect handler", () => {
    test("invokes cleanup helper", async () => {
      validationService.validateRequest.mockReturnValue({
        prompt: "Hello",
        tier: "free",
        ragParams: {},
      });

      cacheService.buildCacheKey.mockReturnValue("key");

      cacheService.tryServeCachedResponse.mockReturnValue(false);

      const task = {
        id: "task-1",
      };

      taskFactory.createGatewayTask.mockReturnValue(task);

      await controller(req, res, next);

      const closeHandler = req.on.mock.calls[0][1];

      closeHandler();

      expect(workerHelpers.cleanupAbortedTask).toHaveBeenCalled();
    });
  });

  // ========================================================
  // Error Handling
  // ========================================================

  describe("error handling", () => {
    test("passes validation errors to next()", async () => {
      const error = new Error("Validation failed");

      validationService.validateRequest.mockImplementation(() => {
        throw error;
      });

      await controller(req, res, next);

      expect(next).toHaveBeenCalledWith(error);
    });

    test("passes task creation errors to next()", async () => {
      validationService.validateRequest.mockReturnValue({
        prompt: "Hello",
        tier: "free",
        ragParams: {},
      });

      cacheService.buildCacheKey.mockReturnValue("key");

      cacheService.tryServeCachedResponse.mockReturnValue(false);

      const error = new Error("Task failure");

      taskFactory.createGatewayTask.mockImplementation(() => {
        throw error;
      });

      await controller(req, res, next);

      expect(next).toHaveBeenCalledWith(error);
    });
  });
});
