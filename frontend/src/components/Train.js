import React, { useState } from 'react';

const Train = () => {
    const [trainingStatus, setTrainingStatus] = useState('');
    const [trainingResult, setTrainingResult] = useState('');

    const initiateTraining = async () => {
        setTrainingStatus('Training in progress...');
        setTrainingResult('');

        try {
            const response = await fetch('/api/train', {
                method: 'POST'
            });
            const data = await response.json();
            setTrainingStatus('Training completed.');
            setTrainingResult(data.message);
        } catch (error) {
            setTrainingStatus('Error during training.');
            setTrainingResult('');
        }
    };

    return (
        <div className="train-container">
            <button onClick={initiateTraining}>Train Model</button>
            <div className="training-status">{trainingStatus}</div>
            <div className="training-result">{trainingResult}</div>
        </div>
    );
};

export default Train;
