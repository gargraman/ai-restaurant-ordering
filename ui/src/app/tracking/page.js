'use client';

import { useState, useEffect } from 'react';
import { Search, Package, Clock, CheckCircle, AlertCircle, Truck } from 'lucide-react';
import { getOrderDetails } from '@/lib/api-client';

export default function OrderTrackingPage() {
  const [orderNumber, setOrderNumber] = useState('');
  const [customerEmail, setCustomerEmail] = useState('');
  const [order, setOrder] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(null);

  // Function to fetch order details
  const fetchOrderDetails = async (orderNum, email) => {
    try {
      const orderData = await getOrderDetails(orderNum, email);
      setOrder(orderData);
      return orderData;
    } catch (err) {
      setError(err.message || 'Failed to retrieve order details');
      setOrder(null);
      return null;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!orderNumber || !customerEmail) {
      setError('Please enter both order number and email');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Request notification permission if not already granted
      if ('Notification' in window && Notification.permission !== 'granted' && Notification.permission !== 'denied') {
        await Notification.requestPermission();
      }

      await fetchOrderDetails(orderNumber, customerEmail);
    } finally {
      setLoading(false);
    }
  };

  // Poll for order status updates when an order is being tracked
  useEffect(() => {
    if (order && !order.status.endsWith('ed') && order.status !== 'canceled' && order.status !== 'failed' && order.status !== 'refunded') {
      // Start polling if order is not in a terminal state
      const interval = setInterval(async () => {
        if (orderNumber && customerEmail) {
          const updatedOrder = await fetchOrderDetails(orderNumber, customerEmail);

          // If status changed, show a notification
          if (updatedOrder && updatedOrder.status !== order.status) {
            // Show status change notification
            console.log(`Order status changed from ${order.status} to ${updatedOrder.status}`);

            // Optionally show a browser notification if supported
            if ('Notification' in window && Notification.permission === 'granted') {
              new Notification('Order Status Updated', {
                body: `Your order status changed from ${order.status.replace('_', ' ')} to ${updatedOrder.status.replace('_', ' ')}`,
                icon: '/favicon.ico'
              });
            }
          }
        }
      }, 30000); // Poll every 30 seconds

      setPollingInterval(interval);
    } else {
      // Clear interval if order is in terminal state
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
    }

    // Cleanup function
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [order, orderNumber, customerEmail]);

  // Function to get status icon based on order status
  const getStatusIcon = (status) => {
    switch (status) {
      case 'created':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      case 'paid':
        return <CheckCircle className="w-5 h-5 text-blue-500" />;
      case 'sent_to_pos':
        return <Package className="w-5 h-5 text-indigo-500" />;
      case 'accepted':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'preparing':
        return <Clock className="w-5 h-5 text-orange-500" />;
      case 'ready':
        return <Truck className="w-5 h-5 text-purple-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'canceled':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'refunded':
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  // Function to get status color based on order status
  const getStatusColor = (status) => {
    switch (status) {
      case 'created':
        return 'bg-yellow-100 text-yellow-800';
      case 'paid':
        return 'bg-blue-100 text-blue-800';
      case 'sent_to_pos':
        return 'bg-indigo-100 text-indigo-800';
      case 'accepted':
        return 'bg-green-100 text-green-800';
      case 'preparing':
        return 'bg-orange-100 text-orange-800';
      case 'ready':
        return 'bg-purple-100 text-purple-800';
      case 'completed':
        return 'bg-green-600 text-white';
      case 'canceled':
        return 'bg-red-100 text-red-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'refunded':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Function to get status description based on order status
  const getStatusDescription = (status) => {
    switch (status) {
      case 'created':
        return 'Order created, awaiting payment';
      case 'paid':
        return 'Payment successful, preparing for POS';
      case 'sent_to_pos':
        return 'Sent to restaurant POS system';
      case 'accepted':
        return 'Order accepted by restaurant, preparing';
      case 'preparing':
        return 'Order is being prepared';
      case 'ready':
        return 'Order is ready for pickup/delivery';
      case 'completed':
        return 'Order completed';
      case 'canceled':
        return 'Order was canceled';
      case 'failed':
        return 'Order failed';
      case 'refunded':
        return 'Order was refunded';
      default:
        return 'Order status unknown';
    }
  };

  // Function to get timeline steps
  const getTimelineSteps = () => {
    const steps = [
      { status: 'created', label: 'Order Placed' },
      { status: 'paid', label: 'Payment Confirmed' },
      { status: 'sent_to_pos', label: 'Sent to Restaurant' },
      { status: 'accepted', label: 'Order Accepted' },
      { status: 'preparing', label: 'Preparing' },
      { status: 'ready', label: 'Ready for Pickup/Delivery' },
      { status: 'completed', label: 'Completed' }
    ];

    // Find the current index
    const currentIndex = steps.findIndex(step => step.status === order?.status);

    return steps.map((step, index) => ({
      ...step,
      isActive: index <= currentIndex,
      isCompleted: index < currentIndex
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Track Your Order</h1>
          <p className="text-gray-600">Enter your order number and email to track your order status</p>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="orderNumber" className="block text-sm font-medium text-gray-700 mb-1">
                  Order Number
                </label>
                <input
                  type="text"
                  id="orderNumber"
                  value={orderNumber}
                  onChange={(e) => setOrderNumber(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                  placeholder="e.g., AB-123456"
                />
              </div>
              <div>
                <label htmlFor="customerEmail" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  type="email"
                  id="customerEmail"
                  value={customerEmail}
                  onChange={(e) => setCustomerEmail(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
                  placeholder="your@email.com"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-orange-500 hover:bg-orange-600 text-white font-medium py-3 px-4 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Tracking...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 mr-2" />
                  Track Order
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center text-red-700">
                <AlertCircle className="w-5 h-5 mr-2" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {order && (
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            {/* Order Header */}
            <div className="bg-gradient-to-r from-orange-500 to-orange-600 p-6 text-white">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-2xl font-bold">Order #{order.order_number}</h2>
                  <p className="opacity-90">Placed on {new Date(order.created_at).toLocaleString()}</p>
                </div>
                <div className="mt-4 md:mt-0">
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(order.status)}`}>
                    {getStatusIcon(order.status)}
                    <span className="ml-2 capitalize">{order.status.replace('_', ' ')}</span>
                  </span>
                </div>
              </div>
            </div>

            {/* Order Status Description */}
            <div className="p-6 border-b">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  {getStatusIcon(order.status)}
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900 capitalize">{order.status.replace('_', ' ')}</h3>
                  <p className="text-gray-600">{getStatusDescription(order.status)}</p>
                </div>
              </div>
            </div>

            {/* Timeline */}
            <div className="p-6 border-b">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Order Progress</h3>
              <div className="space-y-4">
                {getTimelineSteps().map((step, index) => (
                  <div key={step.status} className="flex items-start">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      step.isCompleted ? 'bg-green-500 text-white' : 
                      step.isActive ? 'bg-orange-500 text-white' : 'bg-gray-200 text-gray-500'
                    }`}>
                      {step.isCompleted ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <span className="text-xs font-bold">{index + 1}</span>
                      )}
                    </div>
                    <div className="ml-4">
                      <p className={`font-medium ${
                        step.isActive ? 'text-gray-900' : 'text-gray-500'
                      }`}>
                        {step.label}
                      </p>
                      {step.status === order.status && (
                        <p className="text-sm text-gray-600 mt-1">
                          {getStatusDescription(order.status)}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Order Details */}
            <div className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Order Details</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-md font-medium text-gray-700 mb-2">Items</h4>
                  <ul className="space-y-2">
                    {order.items.map((item, index) => (
                      <li key={index} className="flex justify-between text-sm">
                        <span>
                          {item.quantity}x {item.name}
                          {item.modifiers && item.modifiers.length > 0 && (
                            <span className="text-gray-500"> (+{item.modifiers.length} modifier{item.modifiers.length > 1 ? 's' : ''})</span>
                          )}
                        </span>
                        <span>${(item.total_cents / 100).toFixed(2)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-md font-medium text-gray-700 mb-2">Price Breakdown</h4>
                  <dl className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <dt>Subtotal</dt>
                      <dd>${(order.subtotal_cents / 100).toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Tax</dt>
                      <dd>${(order.tax_cents / 100).toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Tip</dt>
                      <dd>${(order.tip_cents / 100).toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Delivery Fee</dt>
                      <dd>${(order.delivery_fee_cents / 100).toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between pt-2 mt-2 border-t border-gray-200 font-medium">
                      <dt>Total</dt>
                      <dd>${(order.total_cents / 100).toFixed(2)}</dd>
                    </div>
                  </dl>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200">
                <h4 className="text-md font-medium text-gray-700 mb-2">Customer Information</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">Name</p>
                    <p className="font-medium">{order.customer_name || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Email</p>
                    <p className="font-medium">{order.customer_email || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Fulfillment Type</p>
                    <p className="font-medium capitalize">{order.fulfillment_type}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Restaurant ID</p>
                    <p className="font-medium">{order.restaurant_id}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!order && !loading && !error && (
          <div className="bg-white rounded-xl shadow-md p-12 text-center">
            <Package className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Track Your Order</h3>
            <p className="text-gray-600">Enter your order number and email to see the status of your order</p>
          </div>
        )}
      </div>
    </div>
  );
}