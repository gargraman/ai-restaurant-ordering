/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use standalone output only in production for SSR
  // In development, use default output to avoid static asset serving issues
  ...(process.env.NODE_ENV === 'production' && { output: 'standalone' }),
  images: {
    unoptimized: true
  },
  async rewrites() {
    // In Docker: use internal service name (http://api:8000)
    // In development: use localhost (http://localhost:8000)
    // Can be overridden with BACKEND_URL env var
    const backendUrl = process.env.BACKEND_URL || 
      (process.env.NODE_ENV === 'production' ? 'http://api:8000' : 'http://localhost:8000');
    
    console.log(`[Next.js Config] Backend URL: ${backendUrl}`);
    
    return {
      // fallback rewrites only apply when no Next.js page matches the path
      fallback: [
        { source: '/chat/:path*', destination: `${backendUrl}/chat/:path*` },
        { source: '/session/:path*', destination: `${backendUrl}/session/:path*` },
        { source: '/orders/:path*', destination: `${backendUrl}/orders/:path*` },
        { source: '/admin/:path*', destination: `${backendUrl}/admin/:path*` },
        { source: '/health', destination: `${backendUrl}/health` },
        { source: '/metrics', destination: `${backendUrl}/metrics` },
        { source: '/auth/:path*', destination: `${backendUrl}/auth/:path*` },
      ],
    };
  },
}

module.exports = nextConfig