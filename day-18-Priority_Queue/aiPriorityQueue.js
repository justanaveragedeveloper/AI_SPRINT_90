/**
 * @typedef {Object} AiTask
 * @property {string} taskId
 * @property {number} priorityScore
 * @property {string} query
 */

export class AiTaskPriorityQueue {
  constructor() {
    this.heap = [];
    this.sequence = 0; // Used for stable ordering when priorities are equal
  }

  getParentIndex(i) {
    return Math.floor((i - 1) / 2);
  }

  getLeftChildIndex(i) {
    return 2 * i + 1;
  }

  getRightChildIndex(i) {
    return 2 * i + 2;
  }

  swap(i1, i2) {
    [this.heap[i1], this.heap[i2]] = [this.heap[i2], this.heap[i1]];
  }

  /**
   * Compare two tasks.
   * Lower priorityScore wins.
   * If equal, earlier inserted task wins (stable ordering).
   */
  compare(taskA, taskB) {
    if (taskA.priorityScore !== taskB.priorityScore) {
      return taskA.priorityScore - taskB.priorityScore;
    }

    return taskA.sequence - taskB.sequence;
  }

  /**
   * Validate task before insertion.
   * @param {AiTask} task
   */
  validateTask(task) {
    if (typeof task !== "object" || task === null) {
      throw new TypeError("Task must be an object.");
    }

    if (typeof task.taskId !== "string" || task.taskId.trim() === "") {
      throw new TypeError("taskId must be a non-empty string.");
    }

    if (
      typeof task.priorityScore !== "number" ||
      Number.isNaN(task.priorityScore)
    ) {
      throw new TypeError("priorityScore must be a valid number.");
    }

    if (typeof task.query !== "string") {
      throw new TypeError("query must be a string.");
    }
  }

  /**
   * Insert a task into the priority queue.
   * Time Complexity: O(log N)
   * @param {AiTask} task
   */
  enqueue(task) {
    this.validateTask(task);

    this.heap.push({
      ...task,
      sequence: this.sequence++,
    });

    this.heapifyUp(this.heap.length - 1);
  }

  /**
   * View the highest-priority task without removing it.
   * Time Complexity: O(1)
   */
  peek() {
    return this.heap.length === 0
        ? null 
        : this.heap[0];
  }

  /**
   * Remove the highest-priority task.
   * Time Complexity: O(log N)
   */
  dequeue() {
    if (this.heap.length === 0) {
      return null;
    }

    if (this.heap.length === 1) {
      return this.heap.pop();
    }

    const highestPriorityTask = this.heap[0];

    this.heap[0] = this.heap.pop();

    this.heapifyDown(0);

    return highestPriorityTask;
  }

  heapifyUp(index) {
    while (index > 0) {
      const parentIndex = this.getParentIndex(index);

      if (this.compare(this.heap[index], this.heap[parentIndex]) < 0) {
        this.swap(index, parentIndex);
        index = parentIndex;
      } else {
        break;
      }
    }
  }

  heapifyDown(index) {
    while (this.getLeftChildIndex(index) < this.heap.length) {
      let smallestChildIndex = this.getLeftChildIndex(index);
      const rightChildIndex = this.getRightChildIndex(index);

      if (
        rightChildIndex < this.heap.length &&
        this.compare(
          this.heap[rightChildIndex],
          this.heap[smallestChildIndex],
        ) < 0
      ) {
        smallestChildIndex = rightChildIndex;
      }

      if (this.compare(this.heap[smallestChildIndex], this.heap[index]) < 0) {
        this.swap(index, smallestChildIndex);
        index = smallestChildIndex;
      } else {
        break;
      }
    }
  }

  size() {
    return this.heap.length;
  }

  isEmpty() {
    return this.heap.length === 0;
  }
}
