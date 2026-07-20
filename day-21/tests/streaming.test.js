"use strict";

const { PassThrough } = require("stream");
const axios = require("axios");

const { streamPythonResponse } = require("../services/streamService");

jest.mock("axios");

describe("Stream Service", () => {
  let cache;
  let res;
  let abortController;

  beforeEach(() => {
    cache = {
      store: new Map(),

      get(key) {
        return this.store.get(key);
      },

      put(key, value) {
        this.store.set(key, value);
      },
    };

    res = {
      headers: {},
      body: "",
      headersSent: false,

      writeHead(status, headers) {
        this.headersSent = true;
        this.statusCode = status;
        this.headers = {
          ...this.headers,
          ...headers,
        };
      },

      write(chunk) {
        this.body += chunk;
      },

      end() {
        this.ended = true;
      },

      status: jest.fn(function (code) {
        this.statusCode = code;
        return this;
      }),

      json: jest.fn(),
    };

    abortController = new AbortController();

    jest.clearAllMocks();
  });

  // ========================================================
  // Successful Streaming
  // ========================================================

  describe("successful streaming", () => {
    test("streams response and stores it in cache", async () => {
      const stream = new PassThrough();

      axios.mockResolvedValue({
        data: stream,
      });

      const promise = streamPythonResponse({
        res,
        prompt: "Hello",
        ragParams: {},
        cache,
        cacheKey: "cache-key",
        abortController,
      });

      stream.write("Hello ");
      stream.write("World");
      stream.end();

      await promise;

      expect(cache.get("cache-key")).toBe("Hello World");

      expect(res.body).toContain("Hello");
      expect(res.body).toContain("World");

      expect(res.ended).toBe(true);
    });

    test("writes cache miss header", async () => {
      const stream = new PassThrough();

      axios.mockResolvedValue({
        data: stream,
      });

      const promise = streamPythonResponse({
        res,
        prompt: "Hello",
        ragParams: {},
        cache,
        cacheKey: "key",
        abortController,
      });

      stream.end();

      await promise;

      expect(res.headers["X-Cache-Status"]).toBe("MISS");
    });

    test("marks stream complete", async () => {
      const stream = new PassThrough();

      axios.mockResolvedValue({
        data: stream,
      });

      const promise = streamPythonResponse({
        res,
        prompt: "Hello",
        ragParams: {},
        cache,
        cacheKey: "key",
        abortController,
      });

      stream.end();

      await promise;

      expect(res.body).toContain('"done":true');
    });
  });

  // ========================================================
  // Abort
  // ========================================================

  describe("abort", () => {
    test("returns silently after cancellation", async () => {
      abortController.abort();

      axios.mockRejectedValue({
        name: "CanceledError",
      });

      await expect(
        streamPythonResponse({
          res,
          prompt: "Hello",
          ragParams: {},
          cache,
          cacheKey: "key",
          abortController,
        }),
      ).resolves.toBeUndefined();
    });
  });

  // ========================================================
  // Failure
  // ========================================================

  describe("failure", () => {
    test("returns HTTP 500 before headers are sent", async () => {
      axios.mockRejectedValue(new Error("Python unavailable"));

      await streamPythonResponse({
        res,
        prompt: "Hello",
        ragParams: {},
        cache,
        cacheKey: "key",
        abortController,
      });

      expect(res.status).toHaveBeenCalledWith(500);

      expect(res.json).toHaveBeenCalledWith({
        error: "Streaming gateway processing failure.",
      });
    });

    test("writes SSE error after headers are sent", async () => {
      res.headersSent = true;

      axios.mockRejectedValue(new Error("Failure"));

      await expect(
        streamPythonResponse({
          res,
          prompt: "Hello",
          ragParams: {},
          cache,
          cacheKey: "key",
          abortController,
        }),
      ).rejects.toThrow("Failure");

      expect(res.body).toContain("Streaming gateway processing failure.");
    });
  });

  // ========================================================
  // Cache
  // ========================================================

  describe("cache", () => {
    test("stores complete streamed response", async () => {
      const stream = new PassThrough();

      axios.mockResolvedValue({
        data: stream,
      });

      const promise = streamPythonResponse({
        res,
        prompt: "Prompt",
        ragParams: {},
        cache,
        cacheKey: "response",
        abortController,
      });

      stream.write("Part 1 ");
      stream.write("Part 2 ");
      stream.write("Part 3");

      stream.end();

      await promise;

      expect(cache.get("response")).toBe("Part 1 Part 2 Part 3");
    });

    test("does not cache aborted response", async () => {
      const stream = new PassThrough();

      axios.mockResolvedValue({
        data: stream,
      });

      const promise = streamPythonResponse({
        res,
        prompt: "Prompt",
        ragParams: {},
        cache,
        cacheKey: "response",
        abortController,
      });

      stream.write("Partial");

      abortController.abort();

      stream.end();

      await promise;

      expect(cache.get("response")).toBeUndefined();
    });
  });
});
