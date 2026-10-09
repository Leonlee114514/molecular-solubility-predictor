import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

// FastAPI backend for the dev proxy. Override when port 8000 is taken by
// something else:  VITE_API_TARGET=http://localhost:8010 npm run dev
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:8000"

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [inspectAttr(), react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
