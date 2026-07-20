"use strict";

/**
 * ==========================================================
 * Token Estimator
 * ==========================================================
 *
 * Provides a lightweight approximation of the number of
 * LLM tokens contained in a prompt.
 *
 * NOTE:
 * This is NOT an exact tokenizer.
 * It is only used for task scheduling and priority
 * calculation inside the AI Gateway.
 *
 * Production systems often use tokenizer libraries such as:
 * - tiktoken (OpenAI)
 * - sentencepiece
 * - Hugging Face tokenizers
 */

/**
 * Estimates the approximate number of LLM tokens.
 *
 * Strategy:
 * 1. Remove leading/trailing whitespace.
 * 2. Split the text into words.
 * 3. Multiply by 1.3 because English words average
 *    about 1.3 LLM tokens.
 *
 * @param {string} text
 * @returns {number}
 */
function estimateTokenLength(text) {
  if (typeof text !== "string") {
    return 0;
  }

  const trimmedText = text.trim();

  if (trimmedText.length === 0) {
    return 0;
  }

  const wordCount = trimmedText.split(/\s+/).length;

  return Math.ceil(wordCount * 1.3);
}

module.exports = estimateTokenLength;
