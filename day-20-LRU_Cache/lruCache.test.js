// lruCache.test.js
import { AiLruCache } from "./aiLruCache.js";

describe("AI Gateway LRU Cache Barrier Suite", () => {
  it("should store items and maintain strict O(1) eviction invariants", () => {
    const cache = new AiLruCache(3); // Cap capacity tightly at 3 entries

    cache.put("query_1", "Vector response embedding payload alpha");
    cache.put("query_2", "Vector response embedding payload beta");
    cache.put("query_3", "Vector response embedding payload gamma");

    expect(cache.size()).toBe(3);
    expect(cache.get("query_1")).toContain("alpha");

    // Adding a 4th item must push out the Least Recently Used item
    // Since we just read 'query_1', 'query_2' is now the oldest item
    cache.put("query_4", "Vector response embedding payload delta");

    expect(cache.size()).toBe(3);
    expect(cache.get("query_2")).toBeNull(); // Should be cleanly evicted
    expect(cache.get("query_4")).not.toBeNull();
  });

  it("should return null for non-existent hash records gracefully", () => {
    const cache = new AiLruCache(2);
    expect(cache.get("unknown_cache_key")).toBeNull();
  });
});
