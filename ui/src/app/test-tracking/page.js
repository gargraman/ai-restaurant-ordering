'use client';

import TestOrderTracking from '@/components/TestOrderTracking';

export default function TestTrackingPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Test Order Tracking</h1>
          <p className="text-gray-600">Test the order tracking functionality</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-md p-6">
          <TestOrderTracking />
        </div>
      </div>
    </div>
  );
}