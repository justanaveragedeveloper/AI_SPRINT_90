"use strict";

const {
  buildCacheKey,
  getCachedResponse,
  storeResponse,
  tryServeCachedResponse,
} = require("../services/cacheService");

describe("Cache Service", () => {
  let cache;

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
  });

  // ========================================================
  // Cache Key Generation
  // ========================================================

  describe("buildCacheKey()", () => {
    test("returns deterministic keys", () => {
      const key1 = buildCacheKey("Hello", { topK: 5 });

      const key2 = buildCacheKey("Hello", { topK: 5 });

      expect(key1).toBe(key2);
    });

    test("changes when prompt changes", () => {
      const key1 = buildCacheKey("Hello", {});

      const key2 = buildCacheKey("Hi", {});

      expect(key1).not.toBe(key2);
    });

    test("changes when rag parameters change", () => {
      const key1 = buildCacheKey("Hello", { topK: 5 });

      const key2 = buildCacheKey("Hello", { topK: 10 });

      expect(key1).not.toBe(key2);
    });

    test("ignores object key order", () => {
      const key1 = buildCacheKey("Hello", {
        topK: 5,
        temperature: 0.2,
      });

      const key2 = buildCacheKey("Hello", {
        temperature: 0.2,
        topK: 5,
      });

      expect(key1).toBe(key2);
    });
  });

  // ========================================================
  // Cache Storage
  // ========================================================

  describe("storeResponse()", () => {
    test("stores response", () => {
      storeResponse(cache, "abc", "AI Response");

      expect(cache.get("abc")).toBe("AI Response");
    });
  });

  // ========================================================
  // Cache Lookup
  // ========================================================

  describe("getCachedResponse()", () => {
    test("returns cached response", () => {
      cache.put("abc", "cached");

      expect(getCachedResponse(cache, "abc")).toBe("cached");
    });

    test("returns undefined for missing key", () => {
      expect(getCachedResponse(cache, "missing")).toBeUndefined();
    });
  });

  // ========================================================
  // Cache Hit
  // ========================================================

  describe("tryServeCachedResponse()", () => {
    let res;

    beforeEach(() => {
      res = {
        headers: {},
        body: "",

        setHeader(name, value) {
          this.headers[name] = value;
        },

        writeHead(status, headers) {
          this.statusCode = status;
          this.headers = {
            ...this.headers,
            ...headers,
          };
        },

        write(chunk) {
          this.body += chunk;
        },

        end() {},
      };
    });

    test("returns false on cache miss", () => {
      const served = tryServeCachedResponse({
        cache,
        cacheKey: "missing",
        res,
      });

      expect(served).toBe(false);
    });

    test("returns true on cache hit", () => {
      cache.put("key", "cached response");

      const served = tryServeCachedResponse({
        cache,
        cacheKey: "key",
        res,
      });

      expect(served).toBe(true);
    });

    test("sets cache header", () => {
      cache.put("key", "cached response");

      tryServeCachedResponse({
        cache,
        cacheKey: "key",
        res,
      });

      expect(res.headers["X-Cache-Status"]).toBe("HIT");
    });

    test("writes cached response", () => {
      cache.put("key", "cached response");

      tryServeCachedResponse({
        cache,
        cacheKey: "key",
        res,
      });

      expect(res.body).toContain("cached response");
    });

    test("marks stream complete", () => {
      cache.put("key", "cached response");

      tryServeCachedResponse({
        cache,
        cacheKey: "key",
        res,
      });

      expect(res.body).toContain('"done":true');
    });
  });
});
