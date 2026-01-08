/** @type {import('next').NextConfig} */
const nextConfig = {
    rewrites: async () => {
        return [
            {
                source: "/api/:path*",
                destination: "http://api:8000/api/:path*",
            },
            {
                source: "/docs",
                destination: "http://api:8000/docs",
            },
            {
                source: "/openapi.json",
                destination: "http://api:8000/openapi.json",
            },
        ];
    },
};

export default nextConfig;
