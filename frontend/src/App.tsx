import React, { useState, useCallback } from 'react';
import axios from 'axios';

const App: React.FC = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<string>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async () => {
    if (!message.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/chat', { message });
      setChatHistory(prev => [...prev, `User: ${message}`, `Bot: ${response.data.response}`]);
      setMessage('');
    } catch (error) {
      setError('Failed to send message. Please try again.');
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  }, [message]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="container mx-auto p-4 bg-gray-100 min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold mb-6">Peru Leni Jeevi Chat</h1>
      <div className="mb-4 w-full max-w-md">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter your message"
          disabled={isLoading}
          className="border border-gray-300 px-3 py-2 rounded w-full transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          onClick={sendMessage}
          disabled={isLoading}
          className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ml-2 transition duration-300 ease-in-out transform hover:scale-105 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
      {error && (
        <div className="text-red-500 mb-4">{error}</div>
      )}
      <div className="border border-gray-300 p-4 rounded w-full max-w-md bg-white shadow-lg overflow-y-auto max-h-[60vh]">
        {chatHistory.map((item, index) => (
          <p key={index} className="mb-2 animate-fade-in">{item}</p>
        ))}
      </div>
    </div>
  );
};

export default App;
