// aiLruCache.js

class CacheNode {
  constructor(key, value) {
    this.key = key;
    this.value = value;
    this.prev = null;
    this.next = null;
  }
}

export class AiLruCache {
  constructor(capacity = 5) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new Error("Capacity must be a positive integer.");
    }

    this.capacity = capacity;
    this.map = new Map();

    // Basic cache statistics
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;

    // Sentinel nodes simplify insertion/removal logic.
    this.head = new CacheNode(null, null);
    this.tail = new CacheNode(null, null);

    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  /**
   * Returns the cached value.
   * Marks the item as Most Recently Used.
   * Time Complexity: O(1)
   */
  get(key) {
    this.#validateKey(key);

    const node = this.map.get(key);

    if (!node) {
      this.misses++;
      return null;
    }

    this.hits++;
    this.#moveToHead(node);
    return node.value;
  }

  /**
   * Inserts or updates a cache entry.
   * Time Complexity: O(1)
   */
  put(key, value) {
    this.#validateKey(key);

    const existingNode = this.map.get(key);

    if (existingNode) {
      existingNode.value = value;
      this.#moveToHead(existingNode);
      return;
    }

    const newNode = new CacheNode(key, value);

    this.map.set(key, newNode);
    this.#addNode(newNode);

    if (this.map.size > this.capacity) {
      const lruNode = this.tail.prev;

      this.#removeNode(lruNode);
      this.map.delete(lruNode.key);

      this.evictions++;
    }
  }

  /**
   * Removes every entry from the cache.
   */
  clear() {
    this.map.clear();

    this.head.next = this.tail;
    this.tail.prev = this.head;

    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  /**
   * Returns cache statistics.
   */
  getStats() {
    return {
      size: this.size(),
      capacity: this.capacity,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      hitRate:
        this.hits + this.misses === 0
          ? 0
          : this.hits / (this.hits + this.misses),
    };
  }

  size() {
    return this.map.size;
  }

  #validateKey(key) {
    if (key === null || key === undefined) {
      throw new Error("Cache key cannot be null or undefined.");
    }
  }

  #addNode(node) {
    node.prev = this.head;
    node.next = this.head.next;

    this.head.next.prev = node;
    this.head.next = node;
  }

  #removeNode(node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
  }

  #moveToHead(node) {
    this.#removeNode(node);
    this.#addNode(node);
  }
}
