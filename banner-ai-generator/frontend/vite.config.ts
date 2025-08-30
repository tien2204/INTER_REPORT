// frontend/vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Backend configuration
const BACKEND_URL = 'http://172.26.33.210:8000';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 3000,
    host: '0.0.0.0', // Quan trọng: cho phép external connections
    proxy: {
      // API routes
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false,
        timeout: 60000, // Tăng timeout
        headers: {
          'Connection': 'keep-alive'
        }
      },
      // Socket.IO routes - CẢI THIỆN QUAN TRỌNG
      '/socket.io': {
        target: BACKEND_URL,
        ws: true, // Enable WebSocket proxying
        changeOrigin: true,
        secure: false,
        timeout: 0, // No timeout for WebSocket
        // Xử lý WebSocket headers
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('Proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Proxying request:', req.method, req.url);
          });
          proxy.on('proxyReqWs', (proxyReq, req, socket) => {
            console.log('Proxying WebSocket:', req.url);
          });
        }
      }
    },
    // Cấu hình CORS cho dev server
    cors: {
      origin: [
        'http://172.26.33.210:3000',
        'http://localhost:3000',
        'http://172.26.33.210:8000'
      ],
      credentials: true
    }
  },
  define: {
    // Environment variables
    'import.meta.env.VITE_API_URL': JSON.stringify(BACKEND_URL),
    'import.meta.env.VITE_WS_URL': JSON.stringify(BACKEND_URL)
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          socketio: ['socket.io-client']
        }
      }
    }
  },
  // Optimizations
  optimizeDeps: {
    include: ['socket.io-client']
  }
})
