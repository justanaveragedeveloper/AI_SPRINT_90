// routes/aiRoutes.js

import express from "express";
import { handleAiQuery } from "./aiController.js";

const router = express.Router();

router.post("/query", handleAiQuery);

export default router;
