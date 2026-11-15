import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발 중에는 /api 를 앱 서버(8000번)로 넘긴다. 빌드 결과(dist)는 앱 서버가 그대로 내보낸다.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  preview: { port: 4173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", emptyOutDir: true },
});
