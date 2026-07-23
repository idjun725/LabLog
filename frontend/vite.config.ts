import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 기본값(host: 'localhost')이 이 환경에서 IPv6(::1)로만 바인딩돼
  // 127.0.0.1 접속이 거부되는 문제가 있어 IPv4로 명시적으로 고정.
  server: {
    host: '127.0.0.1',
  },
})
