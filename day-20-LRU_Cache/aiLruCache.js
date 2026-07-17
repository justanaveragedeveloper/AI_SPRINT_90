// aiLruCache.js

class CacheNode {
  constructor(key, value) {
    this.key = key;
    this.value = value;
    this.next = null;
    this.prev = null;
  }
}

export class AiLruCache {
  constructor(capacity = 5) {
    this.capacity = capacity;
    this.map = new Map();

    // Establish sentinel head and tail nodes to bypass boundary null checks
    this.head = new CacheNode(null, null);
    this.tail = new CacheNode(null, null);
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  /**
   * Retrieve an entry from the cache barrier
   * Time Complexity: O(1)
   */
  get(key) {
    if (!this.map.has(key)) return null;

    const node = this.map.get(key);
    this._moveToHead(node); // Cache hit: Mark as Most Recently Used
    return node.value;
  }

  /**
   * Write or update an element within the cache
   * Time Complexity: O(1)
   */
  put(key, value) {
    if (this.map.has(key)) {
      const node = this.map.get(key);
      node.value = value;
      this._moveToHead(node);
    } else {
      const newNode = new CacheNode(key, value);
      this.map.set(key, newNode);
      this._addNode(newNode);

      // Eviction Check: If capacity bounds are breached, drop the oldest node
      if (this.map.size > this.capacity) {
        const tailNode = this.tail.prev;
        this._removeNode(tailNode);
        this.map.delete(tailNode.key); // Erase reference map tracking
      }
    }
  }

  // --- High-Performance Pointer Manipulations ---

  _addNode(node) {
    // Always insert immediately after the sentinel head
    node.prev = this.head;
    node.next = this.head.next;
    this.head.next.prev = node;
    this.head.next = node;
  }

  _removeNode(node) {
    const prevNode = node.prev;
    const nextNode = node.next;
    prevNode.next = nextNode;
    nextNode.prev = prevNode;
  }

  _moveToHead(node) {
    this._removeNode(node);
    this._addNode(node);
  }

  size() {
    return this.map.size;
  }
}
