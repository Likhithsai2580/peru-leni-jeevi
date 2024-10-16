import React, { useState, useEffect } from 'react';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    useEffect(() => {
        // Load initial bot info
        const loadBotInfo = async () => {
            try {
                const response = await fetch('/api/bot-info');
                const data = await response.json();
                if (data.modelLoaded) {
                    setMessages([{ sender: 'bot', text: 'Model loaded. You can start chatting!' }]);
                } else {
                    setMessages([{ sender: 'bot', text: 'No model loaded. Please load a model to start chatting.' }]);
                }
            } catch (error) {
                setMessages([{ sender: 'bot', text: 'Error loading bot information.' }]);
            }
        };

        loadBotInfo();
    }, []);

    const sendMessage = async () => {
        if (!input) return;

        const newMessages = [...messages, { sender: 'user', text: input }];
        setMessages(newMessages);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: input })
            });
            const data = await response.json();
            setMessages([...newMessages, { sender: 'bot', text: data.response }]);
        } catch (error) {
            setMessages([...newMessages, { sender: 'bot', text: 'Error sending message.' }]);
        }

        setInput('');
    };

    return (
        <div className="chat-container">
            <div className="chat-box">
                {messages.map((msg, index) => (
                    <div key={index} className={msg.sender === 'bot' ? 'bot-message' : 'user-message'}>
                        {msg.sender === 'bot' ? 'Bot: ' : 'You: '}{msg.text}
                    </div>
                ))}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message here..."
                />
                <button onClick={sendMessage}>Send</button>
            </div>
        </div>
    );
};

export default Chat;
