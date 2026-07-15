// TransactionLog.js

export const TransactionLog = {
  async create(data) {
    console.log("Telemetry Event:", data);
    return true;
  },
};
