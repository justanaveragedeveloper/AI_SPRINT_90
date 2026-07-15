/**
 * Day 15 Automated Test Suite for QueryLog Schema.
 * Validates schema rules without requiring a live MongoDB instance.
 */

const mongoose = require("mongoose");
const { MongoMemoryServer } = require("mongodb-memory-server");
const QueryLog = require("./QueryLog");

let mongoServer;

beforeAll(async () => {
  mongoServer = await MongoMemoryServer.create();

  await mongoose.connect(mongoServer.getUri());
});

afterAll(async () => {
  await mongoose.disconnect();
  await mongoServer.stop();
});

describe("QueryLog Model Unit Test Matrix", () => {
  const createValidPayload = () => ({
    requestId: "req-123456",
    userId: new mongoose.Types.ObjectId(),
    userQuery: "How does SQ8 quantization function?",
    compiledPromptLength: 1250,
    executionStatus: "success",
    latencyMs: 42,
  });

  it("should save a valid query log entry", async () => {
    const savedLog = await new QueryLog(createValidPayload()).save();

    expect(savedLog._id).toBeDefined();
    expect(savedLog.executionStatus).toBe("success");
    expect(savedLog.requestId).toBe("req-123456");
  });

  it("should reject invalid execution status values", async () => {
    const payload = createValidPayload();

    payload.executionStatus = "CORRUPTED_SYSTEM_STATUS";

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should reject negative latency values", async () => {
    const payload = createValidPayload();

    payload.latencyMs = -1;

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should reject negative prompt lengths", async () => {
    const payload = createValidPayload();

    payload.compiledPromptLength = -5;

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should reject missing userId", async () => {
    const payload = createValidPayload();

    delete payload.userId;

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should reject oversized user queries", async () => {
    const payload = createValidPayload();

    payload.userQuery = "a".repeat(1001);

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should reject whitespace-only queries", async () => {
    const payload = createValidPayload();

    payload.userQuery = "        ";

    await expect(new QueryLog(payload).save()).rejects.toThrow(
      mongoose.Error.ValidationError,
    );
  });

  it("should trim surrounding whitespace", async () => {
    const payload = createValidPayload();

    payload.userQuery = "   Hello World   ";

    const savedLog = await new QueryLog(payload).save();

    expect(savedLog.userQuery).toBe("Hello World");
  });

  it("should automatically generate timestamps", async () => {
    const savedLog = await new QueryLog(createValidPayload()).save();

    expect(savedLog.createdAt).toBeDefined();

    expect(savedLog.updatedAt).toBeDefined();
  });

  it("should preserve requestId if supplied", async () => {
    const savedLog = await new QueryLog(createValidPayload()).save();

    expect(savedLog.requestId).toBe("req-123456");
  });
});
