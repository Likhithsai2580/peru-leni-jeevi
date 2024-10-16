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
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Peru Leni Jeevi Chat</h1>
      <div className="mb-4">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter your message"
          className="border border-gray-300 px-3 py-2 rounded w-full"
        />
        <button onClick={sendMessage} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ml-2">
          Send
        </button>
      </div>
      <div className="border border-gray-300 p-4 rounded">
        {chatHistory.map((item, index) => (
          <p key={index} className="mb-2">{item}</p>
        ))}
      </div>
    </div>
  );
};

export default App;
