"use strict";

const {
  validatePrompt,
  validateTier,
  validateRagParams,
  validateRequest,
} = require("../services/validationService");

describe("Validation Service", () => {
  // ========================================================
  // Prompt Validation
  // ========================================================

  describe("validatePrompt()", () => {
    test("accepts a valid prompt", () => {
      expect(validatePrompt("Hello AI")).toBe("Hello AI");
    });

    test("trims surrounding whitespace", () => {
      expect(validatePrompt("   Hello AI   ")).toBe("Hello AI");
    });

    test("rejects empty prompt", () => {
      expect(() => validatePrompt("     ")).toThrow("Prompt cannot be empty.");
    });

    test("rejects non-string prompt", () => {
      expect(() => validatePrompt(123)).toThrow("Prompt must be a string.");
    });

    test("rejects oversized prompt", () => {
      const prompt = "a".repeat(50001);

      expect(() => validatePrompt(prompt)).toThrow(
        "Prompt exceeds maximum allowed size.",
      );
    });
  });

  // ========================================================
  // Tier Validation
  // ========================================================

  describe("validateTier()", () => {
    test("accepts free tier", () => {
      expect(validateTier("free")).toBe("free");
    });

    test("accepts premium tier", () => {
      expect(validateTier("premium")).toBe("premium");
    });

    test("normalizes uppercase values", () => {
      expect(validateTier("PREMIUM")).toBe("premium");
    });

    test("defaults to free", () => {
      expect(validateTier()).toBe("free");
    });

    test("rejects unsupported tier", () => {
      expect(() => validateTier("enterprise")).toThrow(
        'Unsupported tier "enterprise".',
      );
    });
  });

  // ========================================================
  // RAG Parameters
  // ========================================================

  describe("validateRagParams()", () => {
    test("accepts empty object", () => {
      expect(validateRagParams({})).toEqual({});
    });

    test("accepts populated object", () => {
      const params = {
        topK: 5,
        temperature: 0.2,
      };

      expect(validateRagParams(params)).toEqual(params);
    });

    test("rejects arrays", () => {
      expect(() => validateRagParams([])).toThrow(
        "ragParams must be an object.",
      );
    });

    test("rejects null", () => {
      expect(() => validateRagParams(null)).toThrow(
        "ragParams must be an object.",
      );
    });
  });

  // ========================================================
  // Complete Request Validation
  // ========================================================

  describe("validateRequest()", () => {
    test("returns cleaned request", () => {
      const result = validateRequest({
        prompt: " Hello ",
        tier: "PREMIUM",
        ragParams: {
          topK: 3,
        },
      });

      expect(result).toEqual({
        prompt: "Hello",
        tier: "premium",
        ragParams: {
          topK: 3,
        },
      });
    });

    test("uses default values", () => {
      const result = validateRequest({
        prompt: "AI",
      });

      expect(result).toEqual({
        prompt: "AI",
        tier: "free",
        ragParams: {},
      });
    });

    test("throws if prompt is invalid", () => {
      expect(() =>
        validateRequest({
          prompt: "",
        }),
      ).toThrow("Prompt cannot be empty.");
    });

    test("throws if tier is invalid", () => {
      expect(() =>
        validateRequest({
          prompt: "Hello",
          tier: "gold",
        }),
      ).toThrow('Unsupported tier "gold".');
    });

    test("throws if ragParams is invalid", () => {
      expect(() =>
        validateRequest({
          prompt: "Hello",
          ragParams: [],
        }),
      ).toThrow("ragParams must be an object.");
    });
  });
});
