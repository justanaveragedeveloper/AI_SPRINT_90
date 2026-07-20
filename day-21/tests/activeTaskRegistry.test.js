"use strict";

const ActiveTaskRegistry = require("../worker/ActiveTaskRegistry");

describe("ActiveTaskRegistry", () => {
  let registry;

  beforeEach(() => {
    registry = new ActiveTaskRegistry();
  });

  // ========================================================
  // Constructor
  // ========================================================

  test("starts empty", () => {
    expect(registry.size()).toBe(0);
  });

  // ========================================================
  // Register
  // ========================================================

  describe("register()", () => {
    test("registers a task", () => {
      const task = {
        id: "task-1",
        execute: jest.fn(),
      };

      registry.register(task);

      expect(registry.size()).toBe(1);
    });

    test("rejects task without id", () => {
      expect(() => {
        registry.register({});
      }).toThrow("Task must have an id.");
    });
  });

  // ========================================================
  // Lookup
  // ========================================================

  describe("lookup", () => {
    const task = {
      id: "task-1",
      execute: jest.fn(),
    };

    beforeEach(() => {
      registry.register(task);
    });

    test("gets task", () => {
      expect(registry.get("task-1")).toEqual(task);
    });

    test("returns undefined for missing task", () => {
      expect(registry.get("missing")).toBeUndefined();
    });

    test("checks task existence", () => {
      expect(registry.has("task-1")).toBe(true);
      expect(registry.has("missing")).toBe(false);
    });
  });

  // ========================================================
  // Remove
  // ========================================================

  describe("remove()", () => {
    test("removes existing task", () => {
      registry.register({
        id: "task-1",
        execute: jest.fn(),
      });

      expect(registry.remove("task-1")).toBe(true);
      expect(registry.size()).toBe(0);
    });

    test("returns false for missing task", () => {
      expect(registry.remove("missing")).toBe(false);
    });
  });

  // ========================================================
  // Clear
  // ========================================================

  describe("clear()", () => {
    test("clears registry", () => {
      registry.register({
        id: "1",
        execute: jest.fn(),
      });

      registry.register({
        id: "2",
        execute: jest.fn(),
      });

      registry.clear();

      expect(registry.size()).toBe(0);
    });
  });
});
