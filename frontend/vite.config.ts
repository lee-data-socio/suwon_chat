import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발 중에는 /api 를 앱 서버(8000번)로 넘긴다. 빌드 결과(dist)는 앱 서버가 그대로 내보낸다.
// 다른 서버(예: 실시간 모드 서버)에 붙이려면 API_TARGET=http://localhost:8010 npm run dev -- --port 5174
const target = process.env.API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": target } },
  preview: { port: 4173, proxy: { "/api": target } },
  build: { outDir: "dist", emptyOutDir: true },
});
