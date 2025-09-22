import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Forward API calls to the Flask backend on :5000 during dev
      '/start':  { target: 'http://localhost:5000', changeOrigin: true },
      '/status': { target: 'http://localhost:5000', changeOrigin: true }, // optional
    },
  },
})