import './globals.css'
import { Inter } from 'next/font/google'
import { ChatProvider } from '@/contexts/ChatContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Restaurant Discovery Chat',
  description: 'Chat-based restaurant and catering discovery',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ChatProvider>
          {children}
        </ChatProvider>
      </body>
    </html>
  )
}