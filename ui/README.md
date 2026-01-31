# Restaurant Discovery Chat UI

A chat-based restaurant and catering discovery interface that connects to the Hybrid Search v2 backend.

## Features

- Conversational interface for discovering restaurants and catering options
- Natural language processing for intuitive queries
- Visual presentation of restaurant results with key details
- Responsive design for mobile and desktop
- Accessibility features following WCAG AA standards
- Analytics tracking for user interactions
- Loading states, error handling, and empty states

## Tech Stack

- React 18
- Next.js 14
- Tailwind CSS
- Lucide React Icons
- UUID for session management

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the UI directory: `cd ui`
3. Install dependencies: `npm install`
4. Create a `.env.local` file in the `ui` directory with the following content:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running the Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Running the Test Version

To run with mock data instead of connecting to the backend:

```bash
npm run dev
```

Then visit `http://localhost:3000/test`

## Environment Variables

- `NEXT_PUBLIC_API_URL` - The URL of the backend API (default: http://localhost:8000)

## Project Structure

```
ui/
├── src/
│   ├── app/           # Next.js app router pages
│   ├── components/    # Reusable UI components
│   ├── contexts/      # React context providers
│   ├── lib/           # Utilities and API clients
│   └── lib/analytics/ # Analytics tracking utilities
├── public/            # Static assets
├── package.json
├── next.config.js
├── tailwind.config.js
└── README.md
```

## API Integration

The UI communicates with the backend through the following endpoints:

- `POST /chat/search` - Perform a chat search
- `GET /session/:sessionId` - Get session details
- `DELETE /session/:sessionId` - Delete a session
- `POST /session/:sessionId/feedback` - Submit feedback

See `src/lib/api-client.js` for implementation details.

## Analytics

The application tracks the following events:

- Query submissions
- Result clicks
- Zero-result queries
- Errors

See `src/lib/analytics/tracker.js` for implementation details.

## Accessibility

The UI follows WCAG AA standards with:

- Proper semantic HTML
- ARIA attributes
- Keyboard navigation support
- Focus management
- Color contrast compliance

## Responsive Design

The UI is designed to work on:

- Mobile devices (320px and up)
- Tablets (768px and up)
- Desktops (1024px and up)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.