import React from 'react';
import { Inter } from 'next/font/google';
import '@/app/globals.css';

const inter = Inter({ subsets: ['latin'] });

export default function AdminLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          {/* Include any admin-specific layout elements here */}
          {children}
        </div>
      </body>
    </html>
  );
}