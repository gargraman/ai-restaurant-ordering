import './globals.css'
import { Inter } from 'next/font/google'
import { ChatProvider } from '@/contexts/ChatContext'
import { CartProvider } from '@/contexts/CartContext'
import { AuthProvider } from '@/contexts/AuthContext'

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  preload: true,
  variable: '--font-inter'
})

export const metadata = {
  title: 'Restaurant Discovery Chat',
  description: 'Chat-based restaurant and catering discovery',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">
        <AuthProvider>
          <CartProvider>
            <ChatProvider>
              {children}
            </ChatProvider>
          </CartProvider>
        </AuthProvider>
      </body>
    </html>
  )
}