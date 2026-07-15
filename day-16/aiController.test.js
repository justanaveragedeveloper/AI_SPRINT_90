import request from "supertest";
import express from "express";
import nock from "nock";
import { jest } from "@jest/globals";

import aiRouter from "./aiRoutes.js";
import { TransactionLog } from "./TransactionLog.js";

const app = express();

app.use(express.json());
app.use("/api/v1/ai", aiRouter);

describe("AI Streaming Gateway Suite", () => {
  afterEach(() => {
    nock.cleanAll();
    jest.clearAllMocks();
  });

  it("should reject missing query", async () => {
    const response = await request(app).post("/api/v1/ai/query").send({
      userId: "user_123",
    });

    expect(response.statusCode).toBe(400);
  });

  it("should reject invalid types", async () => {
    const response = await request(app).post("/api/v1/ai/query").send({
      query: [],
      userId: {},
    });

    expect(response.statusCode).toBe(400);
  });

  it("should reject empty strings", async () => {
    const response = await request(app).post("/api/v1/ai/query").send({
      query: "",
      userId: "",
    });

    expect(response.statusCode).toBe(400);
  });

  it("should return SSE headers", async () => {
    nock("http://localhost:8000")
      .post("/api/v1/predict/stream")
      .reply(200, "hello world");

    TransactionLog.create = jest.fn().mockResolvedValue(true);

    const response = await request(app).post("/api/v1/ai/query").send({
      query: "hello",
      userId: "user_123",
    });

    expect(response.headers["content-type"]).toContain("text/event-stream");

    expect(response.headers["cache-control"]).toBe("no-cache");
  });

  it("should handle python service failure", async () => {
    nock("http://localhost:8000")
      .post("/api/v1/predict/stream")
      .replyWithError("Connection refused");

    const response = await request(app).post("/api/v1/ai/query").send({
      query: "hello",
      userId: "user_123",
    });

    expect(response.statusCode).toBe(200);
    expect(response.text).toContain("[ERROR]");
  });

  it("should continue if telemetry logging fails", async () => {
    nock("http://localhost:8000")
      .post("/api/v1/predict/stream")
      .reply(200, "hello");

    TransactionLog.create = jest
      .fn()
      .mockRejectedValue(new Error("Mongo down"));

    const response = await request(app).post("/api/v1/ai/query").send({
      query: "hello",
      userId: "user_123",
    });

    expect(response.statusCode).toBe(200);
  });

  it("should reject whitespace-only strings", async () => {
    const response = await request(app).post("/api/v1/ai/query").send({
      query: "   ",
      userId: "   ",
    });

    expect(response.statusCode).toBe(400);
  });
});
