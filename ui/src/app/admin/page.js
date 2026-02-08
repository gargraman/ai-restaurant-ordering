'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { 
  createRestaurant, 
  connectStripe, 
  getStripeStatus,
  connectPos,
  getPosStatus
} from '@/lib/api-client';

export default function AdminDashboard() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [restaurants, setRestaurants] = useState([]);
  const [formData, setFormData] = useState({
    name: '',
    slug: '',
    description: '',
    address_line1: '',
    city: '',
    state: '',
    postal_code: '',
    country: 'US',
    phone: '',
    email: '',
    timezone: 'America/New_York'
  });
  const [stripeForm, setStripeForm] = useState({
    email: '',
    refresh_url: typeof window !== 'undefined' ? `${window.location.origin}/admin/restaurants` : '/admin/restaurants',
    return_url: typeof window !== 'undefined' ? `${window.location.origin}/admin/restaurants` : '/admin/restaurants'
  });
  const [posForm, setPosForm] = useState({
    provider: 'square',
    credentials: {},
    location_id: ''
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Check authentication on mount
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    } else if (user && user.role !== 'PLATFORM_ADMIN' && user.role !== 'RESTAURANT_ADMIN') {
      setMessage('Access denied. Admin privileges required.');
    }
  }, [isAuthenticated, isLoading, user, router]);

  const handleSubmitRestaurant = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');
    
    try {
      const response = await createRestaurant(formData);
      setRestaurants([...restaurants, response]);
      setMessage('Restaurant created successfully!');
      setFormData({
        name: '',
        slug: '',
        description: '',
        address_line1: '',
        city: '',
        state: '',
        postal_code: '',
        country: 'US',
        phone: '',
        email: '',
        timezone: 'America/New_York'
      });
    } catch (error) {
      setMessage(`Error creating restaurant: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStripeConnect = async (restaurantId) => {
    setLoading(true);
    setMessage('');
    
    try {
      const response = await connectStripe(restaurantId, stripeForm);
      // Redirect to Stripe onboarding
      window.location.href = response.onboarding_url;
    } catch (error) {
      setMessage(`Error initiating Stripe onboarding: ${error.message}`);
      setLoading(false);
    }
  };

  const handlePosConnect = async (restaurantId) => {
    setLoading(true);
    setMessage('');
    
    try {
      const response = await connectPos(restaurantId, posForm);
      setMessage(`POS connection initiated for restaurant ${restaurantId}`);
    } catch (error) {
      setMessage(`Error connecting POS: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (isLoading) {
    return <div className="flex justify-center items-center h-screen">Loading...</div>;
  }

  if (!isAuthenticated || (user && user.role !== 'PLATFORM_ADMIN' && user.role !== 'RESTAURANT_ADMIN')) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          Access denied. Admin privileges required.
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <div className="flex items-center space-x-4">
            <span className="text-gray-600">Welcome, {user?.email}</span>
            <button 
              onClick={() => router.push('/logout')}
              className="bg-red-500 hover:bg-red-700 text-white py-2 px-4 rounded"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex">
            <button
              className={`px-4 py-3 font-medium text-sm ${
                activeTab === 'dashboard'
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('dashboard')}
            >
              Dashboard
            </button>
            <button
              className={`px-4 py-3 font-medium text-sm ${
                activeTab === 'restaurants'
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('restaurants')}
            >
              Manage Restaurants
            </button>
            <button
              className={`px-4 py-3 font-medium text-sm ${
                activeTab === 'onboarding'
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('onboarding')}
            >
              Onboarding
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {message && (
          <div className={`mb-4 p-4 rounded-md ${
            message.includes('Error') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
          }`}>
            {message}
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Dashboard Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-medium text-gray-900">Total Restaurants</h3>
                <p className="mt-2 text-3xl font-semibold text-indigo-600">{restaurants.length}</p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-medium text-gray-900">Active Connections</h3>
                <p className="mt-2 text-3xl font-semibold text-indigo-600">0</p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-medium text-gray-900">Pending Onboarding</h3>
                <p className="mt-2 text-3xl font-semibold text-indigo-600">0</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'restaurants' && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Manage Restaurants</h2>
            
            {/* Restaurant Creation Form */}
            <div className="bg-white p-6 rounded-lg shadow mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Restaurant</h3>
              <form onSubmit={handleSubmitRestaurant}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                      Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      required
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="slug" className="block text-sm font-medium text-gray-700">
                      Slug *
                    </label>
                    <input
                      type="text"
                      id="slug"
                      required
                      value={formData.slug}
                      onChange={(e) => setFormData({...formData, slug: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                      Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="phone" className="block text-sm font-medium text-gray-700">
                      Phone
                    </label>
                    <input
                      type="tel"
                      id="phone"
                      value={formData.phone}
                      onChange={(e) => setFormData({...formData, phone: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="address_line1" className="block text-sm font-medium text-gray-700">
                      Address Line 1
                    </label>
                    <input
                      type="text"
                      id="address_line1"
                      value={formData.address_line1}
                      onChange={(e) => setFormData({...formData, address_line1: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="city" className="block text-sm font-medium text-gray-700">
                      City
                    </label>
                    <input
                      type="text"
                      id="city"
                      value={formData.city}
                      onChange={(e) => setFormData({...formData, city: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="state" className="block text-sm font-medium text-gray-700">
                      State
                    </label>
                    <input
                      type="text"
                      id="state"
                      value={formData.state}
                      onChange={(e) => setFormData({...formData, state: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="postal_code" className="block text-sm font-medium text-gray-700">
                      Postal Code
                    </label>
                    <input
                      type="text"
                      id="postal_code"
                      value={formData.postal_code}
                      onChange={(e) => setFormData({...formData, postal_code: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                </div>
                
                <div className="mt-6">
                  <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                    Description
                  </label>
                  <textarea
                    id="description"
                    rows={3}
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                
                <div className="mt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                  >
                    {loading ? 'Creating...' : 'Create Restaurant'}
                  </button>
                </div>
              </form>
            </div>
            
            {/* Restaurant List */}
            <div className="bg-white shadow overflow-hidden sm:rounded-md">
              <ul className="divide-y divide-gray-200">
                {restaurants.map((restaurant) => (
                  <li key={restaurant.id}>
                    <div className="px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-indigo-600 truncate">
                          {restaurant.name}
                        </p>
                        <div className="ml-2 flex-shrink-0 flex">
                          <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            {restaurant.is_active ? 'Active' : 'Inactive'}
                          </p>
                        </div>
                      </div>
                      <div className="mt-2 sm:flex sm:justify-between">
                        <div className="sm:flex">
                          <p className="flex items-center text-sm text-gray-500">
                            {restaurant.city}, {restaurant.state}
                          </p>
                        </div>
                        <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                          <button
                            onClick={() => handleStripeConnect(restaurant.id)}
                            className="mr-4 text-sm text-indigo-600 hover:text-indigo-900"
                          >
                            Connect Stripe
                          </button>
                          <button
                            onClick={() => {
                              setPosForm({...posForm, restaurantId: restaurant.id});
                              setActiveTab('onboarding');
                            }}
                            className="text-sm text-indigo-600 hover:text-indigo-900"
                          >
                            Connect POS
                          </button>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'onboarding' && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Onboarding</h2>
            
            {/* Stripe Onboarding */}
            <div className="bg-white p-6 rounded-lg shadow mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Stripe Connect Onboarding</h3>
              <form className="space-y-4">
                <div>
                  <label htmlFor="stripe-email" className="block text-sm font-medium text-gray-700">
                    Business Email *
                  </label>
                  <input
                    type="email"
                    id="stripe-email"
                    required
                    value={stripeForm.email}
                    onChange={(e) => setStripeForm({...stripeForm, email: e.target.value})}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="refresh-url" className="block text-sm font-medium text-gray-700">
                      Refresh URL
                    </label>
                    <input
                      type="url"
                      id="refresh-url"
                      value={stripeForm.refresh_url}
                      onChange={(e) => setStripeForm({...stripeForm, refresh_url: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="return-url" className="block text-sm font-medium text-gray-700">
                      Return URL
                    </label>
                    <input
                      type="url"
                      id="return-url"
                      value={stripeForm.return_url}
                      onChange={(e) => setStripeForm({...stripeForm, return_url: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    />
                  </div>
                </div>
                
                <div>
                  <button
                    type="button"
                    onClick={() => handleStripeConnect('selected_restaurant_id')} // Would come from context
                    disabled={loading}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                  >
                    {loading ? 'Processing...' : 'Begin Stripe Onboarding'}
                  </button>
                </div>
              </form>
            </div>
            
            {/* POS Connection */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-medium text-gray-900 mb-4">POS Connection</h3>
              <form className="space-y-4">
                <div>
                  <label htmlFor="pos-provider" className="block text-sm font-medium text-gray-700">
                    POS Provider *
                  </label>
                  <select
                    id="pos-provider"
                    required
                    value={posForm.provider}
                    onChange={(e) => setPosForm({...posForm, provider: e.target.value})}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  >
                    <option value="square">Square</option>
                    <option value="toast">Toast</option>
                    <option value="olo">Olo</option>
                    <option value="omnivore">Omnivore</option>
                  </select>
                </div>
                
                <div>
                  <label htmlFor="pos-location-id" className="block text-sm font-medium text-gray-700">
                    Location ID
                  </label>
                  <input
                    type="text"
                    id="pos-location-id"
                    value={posForm.location_id}
                    onChange={(e) => setPosForm({...posForm, location_id: e.target.value})}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                
                <div>
                  <label htmlFor="pos-credentials" className="block text-sm font-medium text-gray-700">
                    Credentials (JSON)
                  </label>
                  <textarea
                    id="pos-credentials"
                    rows={4}
                    value={JSON.stringify(posForm.credentials, null, 2)}
                    onChange={(e) => {
                      try {
                        const parsed = JSON.parse(e.target.value);
                        setPosForm({...posForm, credentials: parsed});
                      } catch (err) {
                        // Invalid JSON, don't update
                      }
                    }}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm font-mono text-xs"
                  />
                </div>
                
                <div>
                  <button
                    type="button"
                    onClick={() => handlePosConnect('selected_restaurant_id')} // Would come from context
                    disabled={loading}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                  >
                    {loading ? 'Connecting...' : 'Connect POS'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}