import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [tailwindcss()],
  appType: 'mpa',
  server: { open: '/Dashboard.html', proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { rollupOptions: { input: 'Dashboard.html' } }
});
