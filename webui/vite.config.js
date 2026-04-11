import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
// EdgeVox web UI build config.
//
// `npm run build` outputs straight into `../edgevox/server/static/` so the
// FastAPI app picks up the SPA without any copy step. The dev server proxies
// `/ws` to a locally-running `edgevox-serve` so React HMR works against a real
// backend.
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    build: {
        outDir: path.resolve(__dirname, "../edgevox/server/static"),
        emptyOutDir: true,
        sourcemap: false,
    },
    server: {
        port: 5173,
        proxy: {
            "/ws": {
                target: "ws://127.0.0.1:8765",
                ws: true,
            },
            "/api": "http://127.0.0.1:8765",
        },
    },
});
