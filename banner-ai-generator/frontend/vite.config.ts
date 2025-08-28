import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Thay đổi địa chỉ backend API và WebSocket tại đây
const BACKEND_URL = 'http://172.26.33.210:8000';
const WEBSOCKET_URL = 'ws://172.26.33.210:8000';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true
      },
      '/socket.io': {
        target: WEBSOCKET_URL,
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ws/, '/ws')
      }
    }
  },
  define: {
    // Tạo biến môi trường mà code có thể truy cập
    'import.meta.env.VITE_API_URL': JSON.stringify(BACKEND_URL),
    'import.meta.env.VITE_WS_URL': JSON.stringify(WEBSOCKET_URL)
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 1000
  }
})
