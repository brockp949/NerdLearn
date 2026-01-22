/** @type {import('next').NextConfig} */

// Use Docker service name in container, localhost for local dev
const API_URL = process.env.API_URL || 'http://localhost:8000';

const nextConfig = {
    rewrites: async () => {
        return [
            {
                source: "/api/:path*",
                destination: `${API_URL}/api/:path*`,
            },
            {
                source: "/docs",
                destination: `${API_URL}/docs`,
            },
            {
                source: "/openapi.json",
                destination: `${API_URL}/openapi.json`,
            },
        ];
    },
};

export default nextConfig;
