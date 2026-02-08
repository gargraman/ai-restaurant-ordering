/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use standalone output for SSR instead of static export
  // This allows dynamic pages and avoids window is not defined errors during build
  output: 'standalone',
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig