"use strict";

const AppError = require("./AppError");

/**
 * ==========================================================
 * Validation Error
 * ==========================================================
 *
 * Represents invalid client input.
 */

class ValidationError extends AppError {
  constructor(message) {
    super(message, 400);
  }
}

module.exports = ValidationError;
