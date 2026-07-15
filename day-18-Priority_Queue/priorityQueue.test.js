import { AiTaskPriorityQueue } from "./aiPriorityQueue.js";

describe("AI Scheduling Priority Queue Suite", () => {
  let pq;

  beforeEach(() => {
    pq = new AiTaskPriorityQueue();
  });

  test("should extract tasks in ascending priority order", () => {
    pq.enqueue({
      taskId: "T1",
      priorityScore: 50,
      query: "Free Tier User",
    });

    pq.enqueue({
      taskId: "T2",
      priorityScore: 10,
      query: "Enterprise User",
    });

    pq.enqueue({
      taskId: "T3",
      priorityScore: 5,
      query: "Urgent System Task",
    });

    pq.enqueue({
      taskId: "T4",
      priorityScore: 100,
      query: "Background Job",
    });

    expect(pq.dequeue().taskId).toBe("T3");
    expect(pq.dequeue().taskId).toBe("T2");
    expect(pq.dequeue().taskId).toBe("T1");
    expect(pq.dequeue().taskId).toBe("T4");
    expect(pq.isEmpty()).toBe(true);
  });

  test("should return null when dequeueing an empty queue", () => {
    expect(pq.dequeue()).toBeNull();
  });

  test("should allow peeking without removing the task", () => {
    pq.enqueue({
      taskId: "T1",
      priorityScore: 5,
      query: "Prompt",
    });

    expect(pq.peek().taskId).toBe("T1");
    expect(pq.size()).toBe(1);
  });

  test("should preserve insertion order for equal priorities", () => {
    pq.enqueue({
      taskId: "A",
      priorityScore: 10,
      query: "First",
    });

    pq.enqueue({
      taskId: "B",
      priorityScore: 10,
      query: "Second",
    });

    pq.enqueue({
      taskId: "C",
      priorityScore: 10,
      query: "Third",
    });

    expect(pq.dequeue().taskId).toBe("A");
    expect(pq.dequeue().taskId).toBe("B");
    expect(pq.dequeue().taskId).toBe("C");
  });

  test("should correctly handle negative priorities", () => {
    pq.enqueue({
      taskId: "A",
      priorityScore: -5,
      query: "Negative",
    });

    pq.enqueue({
      taskId: "B",
      priorityScore: 0,
      query: "Zero",
    });

    expect(pq.dequeue().taskId).toBe("A");
  });

  test("should correctly handle floating-point priorities", () => {
    pq.enqueue({
      taskId: "A",
      priorityScore: 0.8,
      query: "A",
    });

    pq.enqueue({
      taskId: "B",
      priorityScore: 0.2,
      query: "B",
    });

    pq.enqueue({
      taskId: "C",
      priorityScore: 0.5,
      query: "C",
    });

    expect(pq.dequeue().taskId).toBe("B");
    expect(pq.dequeue().taskId).toBe("C");
    expect(pq.dequeue().taskId).toBe("A");
  });

  test("should throw for invalid task objects", () => {
    expect(() => pq.enqueue({})).toThrow(TypeError);

    expect(() =>
      pq.enqueue({
        taskId: "A",
        priorityScore: "High",
        query: "Invalid",
      })
    ).toThrow(TypeError);

    expect(() =>
      pq.enqueue({
        taskId: "",
        priorityScore: 10,
        query: "Invalid",
      })
    ).toThrow(TypeError);
  });

  test("should correctly update queue size", () => {
    expect(pq.size()).toBe(0);

    pq.enqueue({
      taskId: "A",
      priorityScore: 1,
      query: "One",
    });

    expect(pq.size()).toBe(1);

    pq.dequeue();

    expect(pq.size()).toBe(0);
  });
});