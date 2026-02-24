import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Important: Listen on all interfaces
    port: 5173,
    watch: {
      usePolling: true,  // Important for Docker on Windows/Mac
    },
    proxy: {
      '/api': {
        target: 'http://backend:8000',  // Use service name in Docker
        changeOrigin: true,
        secure: false,
      }
    }
  }
})