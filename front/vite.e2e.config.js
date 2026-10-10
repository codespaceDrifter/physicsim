import { defineConfig } from 'vite'

// backend (FastAPI) runs on :8080; the dev server proxies the API + websocket to it
const backend = 'http://localhost:8081'

export default defineConfig({
  server: {
    proxy: {
      '/scenes': { target: backend, changeOrigin: true },
      '/prefabs': { target: backend, changeOrigin: true },
      '/recordings': { target: backend, changeOrigin: true },
      '/health': { target: backend, changeOrigin: true },
      '/ws': { target: 'ws://localhost:8081', ws: true },
    }
  }
})
