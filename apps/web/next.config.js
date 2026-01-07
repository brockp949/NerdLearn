/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  transpilePackages: ['@nerdlearn/db'],
  env: {
    SCHEDULER_API_URL: process.env.SCHEDULER_API_URL || 'http://localhost:8001',
    TELEMETRY_API_URL: process.env.TELEMETRY_API_URL || 'http://localhost:8002',
    INFERENCE_API_URL: process.env.INFERENCE_API_URL || 'http://localhost:8003',
  },
}

module.exports = nextConfig
