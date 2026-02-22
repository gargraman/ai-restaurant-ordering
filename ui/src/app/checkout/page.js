'use client';

import { useState, useEffect } from 'react';
import { CreditCard, MapPin, User, Mail, Phone, ArrowLeft } from 'lucide-react';
import { loadStripe } from '@stripe/stripe-js';
import { useCart } from '@/contexts/CartContext';
import { useRouter } from 'next/navigation';
import { createOrder } from '@/lib/api-client';
import { mockCreateOrder } from '@/lib/mock-cart-api';

// Initialize Stripe
let stripePromise = null;
const getStripe = () => {
  if (!stripePromise) {
    stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY);
  }
  return stripePromise;
};

export default function CheckoutPage() {
  const router = useRouter();
  const { items, total, clearCart } = useCart();
  const [step, setStep] = useState(1); // 1: shipping, 2: payment, 3: confirmation
  const [shippingInfo, setShippingInfo] = useState({
    fullName: '',
    email: '',
    phone: '',
    address: '',
    city: '',
    state: '',
    zipCode: '',
    notes: ''
  });
  const [errors, setErrors] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [orderId, setOrderId] = useState(null);
  const [clientSecret, setClientSecret] = useState(null);
  const [confirmedItems, setConfirmedItems] = useState([]);
  const [confirmedTotal, setConfirmedTotal] = useState(0);
  const [stripeElements, setStripeElements] = useState(null);
  const [paymentElement, setPaymentElement] = useState(null);
  const [stripe, setStripe] = useState(null);

  // Initialize Stripe
  useEffect(() => {
    const initializeStripe = async () => {
      const stripeInstance = await getStripe();
      setStripe(stripeInstance);
    };

    initializeStripe();
  }, []);

  const validateStep1 = () => {
    const newErrors = {};
    if (!shippingInfo.fullName.trim()) newErrors.fullName = 'Full name is required';
    if (!shippingInfo.email.trim()) newErrors.email = 'Email is required';
    if (!/\S+@\S+\.\S+/.test(shippingInfo.email)) newErrors.email = 'Email is invalid';
    if (!shippingInfo.phone.trim()) newErrors.phone = 'Phone is required';
    if (!shippingInfo.address.trim()) newErrors.address = 'Address is required';
    if (!shippingInfo.city.trim()) newErrors.city = 'City is required';
    if (!shippingInfo.state.trim()) newErrors.state = 'State is required';
    if (!shippingInfo.zipCode.trim()) newErrors.zipCode = 'ZIP code is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleContinueToPayment = async () => {
    if (!validateStep1()) return;

    setIsProcessing(true);

    try {
      // Prepare order data
      const orderData = {
        shipping_info: {
          full_name: shippingInfo.fullName,
          email: shippingInfo.email,
          phone: shippingInfo.phone,
          address: shippingInfo.address,
          city: shippingInfo.city,
          state: shippingInfo.state,
          zip_code: shippingInfo.zipCode,
          notes: shippingInfo.notes
        },
        fulfillment_type: 'delivery', // Default to delivery
        tip_amount: 0, // Default to 0, could be added as a field later
        guest_identifier: shippingInfo.email // Use email as guest identifier
      };

      // Create order and get payment intent client secret
      let orderResponse;
      try {
        orderResponse = await createOrder(localStorage.getItem('chatSessionId') || '', orderData);
      } catch (apiError) {
        console.warn('Failed to create order via backend, using mock:', apiError.message);
        // Fall back to mock API
        orderResponse = await mockCreateOrder(localStorage.getItem('chatSessionId') || '', orderData);
      }

      // Extract client secret from payment intent
      if (orderResponse.payment && orderResponse.payment.client_secret) {
        setClientSecret(orderResponse.payment.client_secret);
        localStorage.setItem('currentOrderId', orderResponse.id || orderResponse.order_number);
        setStep(2); // Move to payment step
      } else {
        throw new Error('Failed to get payment details');
      }
    } catch (error) {
      console.error('Order creation failed:', error);
      alert('Failed to initialize payment. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Initialize Payment Element after getting client secret
  useEffect(() => {
    if (!clientSecret || !stripe) return;

    const elements = stripe.elements({ clientSecret });
    setStripeElements(elements);

    const paymentElementOptions = {
      layout: "tabs",
      wallets: {
        applePay: "never",
        googlePay: "never",
      }
    };

    const paymentElem = elements.create("payment", paymentElementOptions);
    paymentElem.mount("#payment-element");
    setPaymentElement(paymentElem);

    return () => {
      paymentElem.destroy();
    };
  }, [clientSecret, stripe]);

  const handlePlaceOrder = async () => {
    if (!stripe || !stripeElements) return;

    setIsProcessing(true);

    try {
      // Confirm the payment using the same elements instance that mounted the PaymentElement
      const { error } = await stripe.confirmPayment({
        elements: stripeElements,
        confirmParams: {
          return_url: `${window.location.origin}/checkout`,
        },
      });

      if (error) {
        console.error('Payment confirmation error:', error);
        alert(`Payment failed: ${error.message || 'An error occurred'}`);
        return;
      }

      // Payment succeeded - capture snapshot before clearing cart
      setOrderId(localStorage.getItem('currentOrderId'));
      setConfirmedItems([...items]);
      setConfirmedTotal(total);
      clearCart();
      setStep(3);
    } catch (error) {
      console.error('Payment processing failed:', error);
      alert('Payment processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGoBack = () => {
    if (step > 1) {
      setStep(step - 1);
    } else {
      router.back(); // Go back to previous page
    }
  };

  const handleFinish = () => {
    router.push('/'); // Return to home page
  };

  if (items.length === 0 && step !== 3) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="max-w-3xl mx-auto px-4">
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Your Cart is Empty</h2>
            <p className="text-gray-600 mb-6">There are no items in your cart to checkout.</p>
            <button
              onClick={() => router.push('/')}
              className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg"
            >
              Browse Restaurants
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4">
        {/* Progress indicator */}
        <div className="mb-8">
          <div className="flex justify-between relative">
            <div className="absolute top-4 left-0 right-0 h-0.5 bg-gray-200 -z-10">
              <div
                className="h-full bg-orange-500 transition-all duration-300 ease-in-out"
                style={{ width: `${(step / 3) * 100}%` }}
              ></div>
            </div>
            {[1, 2, 3].map((num) => (
              <div key={num} className="flex flex-col items-center relative z-10">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  step >= num ? 'bg-orange-500 text-white' : 'bg-white border-2 border-gray-300 text-gray-400'
                }`}>
                  {num}
                </div>
                <span className={`mt-2 text-sm ${step >= num ? 'text-orange-600 font-medium' : 'text-gray-500'}`}>
                  {num === 1 ? 'Shipping' : num === 2 ? 'Payment' : 'Confirmation'}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          {/* Step 1: Shipping Information */}
          {step === 1 && (
            <div className="p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                <MapPin className="mr-2 h-5 w-5" />
                Shipping Information
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Full Name *
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      type="text"
                      value={shippingInfo.fullName}
                      onChange={(e) => setShippingInfo({...shippingInfo, fullName: e.target.value})}
                      className={`pl-10 w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                        errors.fullName ? 'border-red-500' : 'border-gray-300'
                      }`}
                      placeholder="John Doe"
                    />
                  </div>
                  {errors.fullName && <p className="mt-1 text-sm text-red-600">{errors.fullName}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Email *
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      type="email"
                      value={shippingInfo.email}
                      onChange={(e) => setShippingInfo({...shippingInfo, email: e.target.value})}
                      className={`pl-10 w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                        errors.email ? 'border-red-500' : 'border-gray-300'
                      }`}
                      placeholder="john@example.com"
                    />
                  </div>
                  {errors.email && <p className="mt-1 text-sm text-red-600">{errors.email}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Phone *
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      type="tel"
                      value={shippingInfo.phone}
                      onChange={(e) => setShippingInfo({...shippingInfo, phone: e.target.value})}
                      className={`pl-10 w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                        errors.phone ? 'border-red-500' : 'border-gray-300'
                      }`}
                      placeholder="(123) 456-7890"
                    />
                  </div>
                  {errors.phone && <p className="mt-1 text-sm text-red-600">{errors.phone}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    ZIP Code *
                  </label>
                  <input
                    type="text"
                    value={shippingInfo.zipCode}
                    onChange={(e) => setShippingInfo({...shippingInfo, zipCode: e.target.value})}
                    className={`w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                      errors.zipCode ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="12345"
                  />
                  {errors.zipCode && <p className="mt-1 text-sm text-red-600">{errors.zipCode}</p>}
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Address *
                  </label>
                  <input
                    type="text"
                    value={shippingInfo.address}
                    onChange={(e) => setShippingInfo({...shippingInfo, address: e.target.value})}
                    className={`w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                      errors.address ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="123 Main St"
                  />
                  {errors.address && <p className="mt-1 text-sm text-red-600">{errors.address}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    City *
                  </label>
                  <input
                    type="text"
                    value={shippingInfo.city}
                    onChange={(e) => setShippingInfo({...shippingInfo, city: e.target.value})}
                    className={`w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                      errors.city ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="New York"
                  />
                  {errors.city && <p className="mt-1 text-sm text-red-600">{errors.city}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    State *
                  </label>
                  <input
                    type="text"
                    value={shippingInfo.state}
                    onChange={(e) => setShippingInfo({...shippingInfo, state: e.target.value})}
                    className={`w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500 ${
                      errors.state ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="NY"
                  />
                  {errors.state && <p className="mt-1 text-sm text-red-600">{errors.state}</p>}
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Special Instructions (Optional)
                  </label>
                  <textarea
                    value={shippingInfo.notes}
                    onChange={(e) => setShippingInfo({...shippingInfo, notes: e.target.value})}
                    className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-orange-500"
                    rows="3"
                    placeholder="Delivery instructions, gate codes, etc."
                  />
                </div>
              </div>

              <div className="mt-8 flex justify-between">
                <button
                  onClick={handleGoBack}
                  className="flex items-center text-gray-600 hover:text-gray-800 font-medium"
                  disabled={isProcessing}
                >
                  <ArrowLeft className="mr-2 h-5 w-5" />
                  Back
                </button>
                <button
                  onClick={handleContinueToPayment}
                  disabled={isProcessing}
                  className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg disabled:opacity-50"
                >
                  {isProcessing ? 'Processing...' : 'Continue to Payment'}
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Payment Information */}
          {step === 2 && (
            <div className="p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                <CreditCard className="mr-2 h-5 w-5" />
                Payment Information
              </h2>

              <div className="mb-6">
                <div id="payment-element" className="border rounded-lg p-4"></div>
              </div>

              <div className="mt-8 flex justify-between">
                <button
                  onClick={handleGoBack}
                  className="flex items-center text-gray-600 hover:text-gray-800 font-medium"
                  disabled={isProcessing}
                >
                  <ArrowLeft className="mr-2 h-5 w-5" />
                  Back
                </button>
                <button
                  onClick={handlePlaceOrder}
                  disabled={isProcessing || !stripeElements}
                  className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg disabled:opacity-50"
                >
                  {isProcessing ? 'Processing...' : `Pay $${total.toFixed(2)}`}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Confirmation */}
          {step === 3 && (
            <div className="p-8 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Order Confirmed!</h2>
              <p className="text-gray-600 mb-6">
                Thank you for your order. Your order number is <span className="font-semibold">#{orderId || `ORD-${Math.floor(Math.random() * 1000000)}`}</span>.
                A confirmation email has been sent to {shippingInfo.email}.
              </p>

              <div className="bg-gray-50 rounded-lg p-4 mb-6 text-left">
                <h3 className="font-medium text-gray-800 mb-2">Order Summary</h3>
                <div className="space-y-2">
                  {confirmedItems.map((item, index) => (
                    <div key={index} className="flex justify-between text-sm">
                      <span>{item.quantity}x {item.name}</span>
                      <span>${(item.price * item.quantity).toFixed(2)}</span>
                    </div>
                  ))}
                  <div className="border-t border-gray-200 pt-2 mt-2">
                    <div className="flex justify-between font-medium">
                      <span>Total</span>
                      <span>${confirmedTotal.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <p className="text-gray-600 mb-3">
                  Your order has been placed successfully! You can track your order status using the order number.
                </p>
                <a
                  href={`/tracking?orderNumber=${orderId || `ORD-${Math.floor(Math.random() * 1000000)}`}&email=${shippingInfo.email}`}
                  className="inline-flex items-center text-orange-600 hover:text-orange-800 font-medium"
                >
                  Track your order
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </a>
              </div>

              <button
                onClick={handleFinish}
                className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg"
              >
                Continue Shopping
              </button>
            </div>
          )}
        </div>

        {/* Order Summary Sidebar */}
        {step !== 3 && (
          <div className="mt-6 bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-medium text-gray-800 mb-4">Order Summary</h3>
            <div className="space-y-4">
              {items.map((item, index) => (
                <div key={index} className="flex justify-between items-center border-b pb-3 last:border-0 last:pb-0">
                  <div>
                    <p className="font-medium text-gray-800">{item.name}</p>
                    <p className="text-sm text-gray-600">{item.restaurant_name}</p>
                  </div>
                  <p className="font-medium">${(item.price * item.quantity).toFixed(2)}</p>
                </div>
              ))}
            </div>
            <div className="border-t border-gray-200 pt-4 mt-4">
              <div className="flex justify-between text-lg font-bold">
                <span>Total</span>
                <span>${total.toFixed(2)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}