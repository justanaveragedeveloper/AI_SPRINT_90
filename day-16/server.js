import express from "express";
import cors from "cors";
import aiRouter from "./aiRoutes.js";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.use("/api/v1/ai", aiRouter);

app.use((err, req, res, next) => {
  console.error(err);

  if (!res.headersSent) {
    return res.status(500).json({
      success: false,
      error: "Internal server error",
    });
  }

  next(err);
});

app.listen(PORT, () => {
  console.log(`AI Gateway running on port ${PORT}`);
});
