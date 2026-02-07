import { useState } from 'react';
import { getOrderDetails } from '@/lib/api-client';

export default function TestOrderTracking() {
  const [orderNumber, setOrderNumber] = useState('AB-000001');
  const [customerEmail, setCustomerEmail] = useState('test@example.com');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const testLookup = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getOrderDetails(orderNumber, customerEmail);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold mb-4">Test Order Tracking API</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Order Number</label>
          <input
            type="text"
            value={orderNumber}
            onChange={(e) => setOrderNumber(e.target.value)}
            className="w-full p-2 border rounded"
            placeholder="Enter order number"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Customer Email</label>
          <input
            type="email"
            value={customerEmail}
            onChange={(e) => setCustomerEmail(e.target.value)}
            className="w-full p-2 border rounded"
            placeholder="Enter customer email"
          />
        </div>
        
        <button
          onClick={testLookup}
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Test Lookup'}
        </button>
        
        {error && (
          <div className="p-2 bg-red-100 text-red-700 rounded">
            Error: {error}
          </div>
        )}
        
        {result && (
          <div className="p-4 bg-green-50 border rounded">
            <h3 className="font-bold mb-2">Success! Order Details:</h3>
            <pre className="text-sm overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}