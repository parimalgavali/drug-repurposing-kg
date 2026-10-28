import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

const api = process.env.API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [vue()],
  server: { proxy: { "/api": api } },
  preview: { proxy: { "/api": api } },
});
