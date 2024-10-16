import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Chat from './components/Chat';
import Train from './components/Train';

const App = () => {
  return (
    <Router>
      <div className="App">
        <Switch>
          <Route path="/chat" component={Chat} />
          <Route path="/train" component={Train} />
        </Switch>
      </div>
    </Router>
  );
};

export default App;
