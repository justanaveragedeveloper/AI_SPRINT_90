"use strict";

/**
 * ==========================================================
 * Worker Helper Utilities
 * ==========================================================
 *
 * Helpers related to task cancellation.
 *
 * This file does NOT manage workers.
 */

function cleanupAbortedTask({
  gatewayQueue,
  activeTaskRegistry,
  task,
  abortController,
}) {
  if (abortController.signal.aborted) {
    return;
  }

  abortController.abort();

  gatewayQueue.removeTaskById(task.id);

  activeTaskRegistry.remove(task.id);
}

module.exports = {
  cleanupAbortedTask,
};
