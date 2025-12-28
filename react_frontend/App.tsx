
import React, { useState } from 'react';
import { AuthForm } from './components/AuthForm';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { UserProfile } from './types';

const App: React.FC = () => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [view, setView] = useState<'login' | 'signup'>('login');

  const handleToggleView = () => {
    setView(prev => prev === 'login' ? 'signup' : 'login');
  };

  const handleLogin = (profile: UserProfile) => {
    setUser(profile);
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (user) {
    return <Dashboard user={user} onLogout={handleLogout} />;
  }

  return (
    <Layout>
      <div className="w-full max-w-md">
        <AuthForm 
          mode={view} 
          onToggle={handleToggleView} 
          onSuccess={handleLogin}
        />
      </div>
    </Layout>
  );
};

export default App;
