import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/agent": {
        target: backend,
        changeOrigin: true,
      },
      "/documents": {
        target: backend,
        changeOrigin: true,
      },
      "/health": {
        target: backend,
        changeOrigin: true,
      },
    },
  },
});
