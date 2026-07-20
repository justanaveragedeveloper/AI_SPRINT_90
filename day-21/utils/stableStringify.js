"use strict";

/**
 * ==========================================================
 * Stable JSON Serializer
 * ==========================================================
 *
 * JSON.stringify() depends on the order of object keys.
 * This utility sorts object keys recursively before
 * serializing them, ensuring deterministic output.
 *
 * Example:
 *
 * { a: 1, b: 2 }
 *
 * and
 *
 * { b: 2, a: 1 }
 *
 * will always produce the same string.
 */

/**
 * Recursively serializes any JavaScript value into a
 * deterministic JSON string.
 *
 * @param {*} value
 * @returns {string}
 */
function stableStringify(value) {
  // Primitive values
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  // Arrays (preserve order)
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }

  // Objects (sort keys)
  const sortedKeys = Object.keys(value).sort();

  const serializedPairs = sortedKeys.map((key) => {
    return `${JSON.stringify(key)}:${stableStringify(value[key])}`;
  });

  return `{${serializedPairs.join(",")}}`;
}

module.exports = stableStringify;
