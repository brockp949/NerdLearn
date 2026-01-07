/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["@nerdlearn/ui"],
  reactStrictMode: true,
  images: {
    domains: ["localhost"],
  },
};

module.exports = nextConfig;
