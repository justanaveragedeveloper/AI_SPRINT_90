"use strict";

const { PassThrough } = require("stream");
const request = require("supertest");
const nock = require("nock");

const app = require("../app");

describe("AI Gateway Integration", () => {
  afterEach(() => {
    nock.cleanAll();
  });

  afterAll(() => {
    nock.restore();
  });

  // =======================================================
  // Health
  // =======================================================

  test("GET /health returns healthy", async () => {
    const response = await request(app).get("/health");

    expect(response.status).toBe(200);

    expect(response.body).toEqual({
      status: "healthy",
    });
  });

  // =======================================================
  // Validation
  // =======================================================

  test("rejects invalid request", async () => {
    const response = await request(app).post("/api/ai/generate").send({});

    expect(response.status).toBe(400);
  });

  // =======================================================
  // Successful Stream
  // =======================================================

  test("streams successful AI response", async () => {
    const stream = new PassThrough();

    stream.end("Hello AI");

    nock("http://localhost:5001")
      .post("/stream")
      .reply(200, () => stream);

    const response = await request(app).post("/api/ai/generate").send({
      prompt: "Hello",
    });

    expect(response.status).toBe(200);
  });

  // =======================================================
  // Cache
  // =======================================================

  test("serves cached response on second request", async () => {
    const stream = new PassThrough();

    stream.end("Cached Response");

    nock("http://localhost:5001")
      .post("/stream")
      .once()
      .reply(200, () => stream);

    await request(app).post("/api/ai/generate").send({
      prompt: "Hello",
    });

    const response = await request(app).post("/api/ai/generate").send({
      prompt: "Hello",
    });

    expect(response.status).toBe(200);

    expect(response.headers["x-cache-status"]).toBe("HIT");
  });

  // =======================================================
  // Python Failure
  // =======================================================

  test("returns gateway failure if python service fails", async () => {
    nock("http://localhost:5001")
      .post("/stream")
      .replyWithError("Python Offline");

    const response = await request(app).post("/api/ai/generate").send({
      prompt: "Hello",
    });

    expect(response.status).toBe(500);
  });
});
