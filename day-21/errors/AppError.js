"use strict";

/**
 * ==========================================================
 * Base Application Error
 * ==========================================================
 *
 * Parent class for all custom application errors.
 */

class AppError extends Error {
  constructor(message, statusCode = 500) {
    super(message);

    this.name = this.constructor.name;
    this.statusCode = statusCode;

    Error.captureStackTrace?.(this, this.constructor);
  }
}

module.exports = AppError;
