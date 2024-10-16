import React, { useState } from 'react';
import axios from 'axios';

const App: React.FC = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<string>>([]);

  const sendMessage = async () => {
    try {
      const response = await axios.post('/chat', { message });
      setChatHistory([...chatHistory, `User: ${message}`, `Bot: ${response.data.response}`]);
      setMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
      setChatHistory([...chatHistory, 'Error sending message.']);
    }
  };

  return (
    <div className="container mx-auto p-4 bg-gray-100 min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold mb-6 animate-bounce">Peru Leni Jeevi Chat</h1>
      <div className="mb-4 w-full max-w-md">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter your message"
          className="border border-gray-300 px-3 py-2 rounded w-full transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button onClick={sendMessage} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ml-2 transition duration-300 ease-in-out transform hover:scale-105">
          Send
        </button>
      </div>
      <div className="border border-gray-300 p-4 rounded w-full max-w-md bg-white shadow-lg">
        {chatHistory.map((item, index) => (
          <p key={index} className="mb-2 animate-fade-in">{item}</p>
        ))}
      </div>
    </div>
  );
};

export default App;
