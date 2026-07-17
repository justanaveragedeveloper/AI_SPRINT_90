import { AiLruCache } from "./aiLruCache.js";

describe("AiLruCache", () => {
  test("stores and retrieves values", () => {
    const cache = new AiLruCache(3);

    cache.put("A", 1);
    cache.put("B", 2);

    expect(cache.get("A")).toBe(1);
    expect(cache.get("B")).toBe(2);
    expect(cache.size()).toBe(2);
  });

  test("evicts the least recently used item", () => {
    const cache = new AiLruCache(3);

    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);

    cache.get("A");

    cache.put("D", 4);

    expect(cache.get("B")).toBeNull();
    expect(cache.get("A")).toBe(1);
    expect(cache.get("C")).toBe(3);
    expect(cache.get("D")).toBe(4);
  });

  test("updates existing key without increasing cache size", () => {
    const cache = new AiLruCache(2);

    cache.put("A", 1);
    cache.put("A", 100);

    expect(cache.get("A")).toBe(100);
    expect(cache.size()).toBe(1);
  });

  test("works correctly with capacity one", () => {
    const cache = new AiLruCache(1);

    cache.put("A", 1);
    cache.put("B", 2);

    expect(cache.get("A")).toBeNull();
    expect(cache.get("B")).toBe(2);
  });

  test("returns null for missing keys", () => {
    const cache = new AiLruCache(2);

    expect(cache.get("missing")).toBeNull();
  });

  test("throws error for invalid capacity", () => {
    expect(() => new AiLruCache(0)).toThrow();
    expect(() => new AiLruCache(-1)).toThrow();
    expect(() => new AiLruCache(1.5)).toThrow();
  });

  test("throws error for invalid keys", () => {
    const cache = new AiLruCache(2);

    expect(() => cache.put(null, 1)).toThrow();
    expect(() => cache.put(undefined, 1)).toThrow();

    expect(() => cache.get(null)).toThrow();
    expect(() => cache.get(undefined)).toThrow();
  });

  test("clear removes all cached entries", () => {
    const cache = new AiLruCache(3);

    cache.put("A", 1);
    cache.put("B", 2);

    cache.clear();

    expect(cache.size()).toBe(0);
    expect(cache.getStats().hits).toBe(0);
    expect(cache.getStats().misses).toBe(0);
    expect(cache.getStats().evictions).toBe(0);
  });

  test("tracks cache statistics correctly", () => {
    const cache = new AiLruCache(2);

    cache.put("A", 1);
    cache.put("B", 2);

    cache.get("A");
    cache.get("missing");

    cache.put("C", 3);

    const stats = cache.getStats();

    expect(stats.hits).toBe(1);
    expect(stats.misses).toBe(1);
    expect(stats.evictions).toBe(1);
    expect(stats.size).toBe(2);
  });

  test("cache size never exceeds capacity", () => {
    const cache = new AiLruCache(5);

    for (let i = 0; i < 1000; i++) {
      cache.put(`key-${i}`, i);
    }

    expect(cache.size()).toBe(5);
  });
});
