// workerPool.test.js
import { AiWorkerPool } from "./aiWorkerPool.js";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

describe("AiWorkerPool", () => {
  it("never exceeds maximum concurrency", async () => {
    const pool = new AiWorkerPool(2);

    let running = 0;
    let maxRunning = 0;

    const workload = async () => {
      running++;
      maxRunning = Math.max(maxRunning, running);

      await wait(30);

      running--;
    };

    for (let i = 0; i < 6; i++) {
      pool.submitTask(
        {
          taskId: `${i}`,
          priorityScore: i,
        },
        workload,
      );
    }

    await wait(120);

    expect(maxRunning).toBe(2);
    expect(pool.getActiveCount()).toBe(0);
  });

  it("continues processing after a task throws", async () => {
    const pool = new AiWorkerPool(1);

    const completed = [];

    pool.submitTask(
      {
        taskId: "bad",
        priorityScore: 100,
      },
      async () => {
        throw new Error("failure");
      },
    );

    pool.submitTask(
      {
        taskId: "good",
        priorityScore: 50,
      },
      async () => {
        completed.push("good");
      },
    );

    await wait(50);

    expect(completed).toEqual(["good"]);
    expect(pool.getActiveCount()).toBe(0);
  });

  it("rejects invalid maxConcurrency", () => {
    expect(() => new AiWorkerPool(0)).toThrow();
    expect(() => new AiWorkerPool(-1)).toThrow();
    expect(() => new AiWorkerPool(1.5)).toThrow();
  });

  it("rejects invalid execution function", () => {
    const pool = new AiWorkerPool();

    expect(() =>
      pool.submitTask(
        {
          taskId: "A",
          priorityScore: 10,
        },
        null,
      ),
    ).toThrow();
  });

  it("rejects invalid task object", () => {
    const pool = new AiWorkerPool();

    expect(() => pool.submitTask({}, async () => {})).toThrow();

    expect(() =>
      pool.submitTask(
        {
          taskId: "",
          priorityScore: 1,
        },
        async () => {},
      ),
    ).toThrow();

    expect(() =>
      pool.submitTask(
        {
          taskId: "A",
          priorityScore: "high",
        },
        async () => {},
      ),
    ).toThrow();
  });
});
