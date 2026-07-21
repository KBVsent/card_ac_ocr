import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import CardScanner from "./CardScanner";
import "./styles.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("缺少应用挂载节点 #root");
}

createRoot(root).render(
  <StrictMode>
    <CardScanner />
  </StrictMode>,
);
