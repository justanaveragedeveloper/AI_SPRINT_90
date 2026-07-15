/**
 * Day 15 Production Mongoose Schema: User Query Analytics Log.
 * Engineered for high-throughput append-only operations and efficient analytical index patterns.
 */

const mongoose = require("mongoose");

const QueryLogSchema = new mongoose.Schema(
  {
    requestId: {
      type: String,
      trim: true,
      index: true,
    },

    userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: [true, "userId is required"],
      ref: "User",
    },

    userQuery: {
      type: String,
      required: [true, "userQuery is required"],
      trim: true,
      minlength: [1, "userQuery cannot be empty"],
      maxlength: [1000, "userQuery exceeds maximum length of 1000 characters"],
      validate: {
        validator: function (value) {
          return value.trim().length > 0;
        },
        message: "userQuery cannot contain only whitespace",
      },
    },

    compiledPromptLength: {
      type: Number,
      required: [true, "compiledPromptLength is required"],
      min: [0, "compiledPromptLength cannot be negative"],
      validate: {
        validator: Number.isInteger,
        message: "compiledPromptLength must be an integer",
      },
    },

    executionStatus: {
      type: String,
      required: [true, "executionStatus is required"],
      enum: {
        values: ["success", "failure", "insufficient_context"],
        message: "Invalid execution status",
      },
      default: "success",
    },

    latencyMs: {
      type: Number,
      required: [true, "latencyMs is required"],
      min: [0, "latencyMs cannot be negative"],
      validate: {
        validator: Number.isInteger,
        message: "latencyMs must be an integer",
      },
    },
  },
  {
    timestamps: true,
  },
);

// Optimizes chronological lookups for specific users
QueryLogSchema.index({ userId: 1, createdAt: -1 });

// Optimizes operational health dashboards
QueryLogSchema.index({ executionStatus: 1 });

module.exports = mongoose.model("QueryLog", QueryLogSchema);
