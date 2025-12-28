
import React, { useState } from 'react';
import { User, Lock, ArrowRight, Loader2, ChevronDown, ShieldCheck } from 'lucide-react';
import { UserProfile, Role } from '../types';

interface AuthFormProps {
  mode: 'login' | 'signup';
  onToggle: () => void;
  onSuccess: (profile: UserProfile) => void;
}

export const AuthForm: React.FC<AuthFormProps> = ({ mode, onToggle, onSuccess }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    role: 'admin' as Role
  });
  const [isRoleOpen, setIsRoleOpen] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 800));
    setIsLoading(false);
    onSuccess({
      username: formData.username,
      role: formData.role
    });
  };

  const roles: Role[] = ['admin', 'doctor', 'nurse', 'patient'];

  return (
    <div className="bg-white rounded-[2.5rem] shadow-[0_20px_60px_rgba(0,0,0,0.03)] border border-slate-100 p-10 w-full animate-in fade-in zoom-in-95 duration-500">
      <div className="mb-10 text-center">
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">
          {mode === 'login' ? 'Welcome Back' : 'Create Account'}
        </h2>
        <p className="text-slate-400 mt-2 text-sm font-medium">
          {mode === 'login' 
            ? 'Sign in to your account' 
            : 'Get started for free'}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6 flex flex-col items-stretch w-full">
        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[10px] uppercase tracking-[0.2em] font-black text-slate-400 ml-1">
              Username
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-300 group-focus-within:text-slate-900 transition-colors">
                <User size={16} />
              </div>
              <input
                type="text"
                required
                className="block w-full pl-11 pr-4 py-3.5 bg-slate-50 border border-slate-100 rounded-2xl text-slate-900 text-sm focus:outline-none focus:border-slate-400 transition-all placeholder:text-slate-300 font-medium"
                placeholder="Your username"
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] uppercase tracking-[0.2em] font-black text-slate-400 ml-1">
              Password
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-300 group-focus-within:text-slate-900 transition-colors">
                <Lock size={16} />
              </div>
              <input
                type="password"
                required
                className="block w-full pl-11 pr-4 py-3.5 bg-slate-50 border border-slate-100 rounded-2xl text-slate-900 text-sm focus:outline-none focus:border-slate-400 transition-all placeholder:text-slate-300 font-medium"
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
              />
            </div>
          </div>

          {mode === 'signup' && (
            <div className="space-y-1.5 animate-in fade-in slide-in-from-top-2 duration-300">
              <label className="text-[10px] uppercase tracking-[0.2em] font-black text-slate-400 ml-1">
                Your Role
              </label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setIsRoleOpen(!isRoleOpen)}
                  className="w-full flex items-center justify-between pl-4 pr-4 py-3.5 bg-slate-50 border border-slate-100 rounded-2xl text-slate-900 text-sm transition-all text-left font-bold capitalize hover:bg-white hover:border-slate-300 shadow-sm focus:outline-none"
                >
                  <div className="flex items-center gap-2">
                    <ShieldCheck size={16} className="text-slate-900" />
                    {formData.role}
                  </div>
                  <ChevronDown size={16} className="text-slate-300" />
                </button>
                {isRoleOpen && (
                  <div className="absolute z-20 top-full mt-2 w-full bg-white border border-slate-100 rounded-2xl shadow-2xl overflow-hidden py-1.5 animate-in fade-in slide-in-from-top-1">
                    {roles.map(r => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => { setFormData({...formData, role: r}); setIsRoleOpen(false); }}
                        className="w-full px-4 py-3 text-left text-xs font-bold text-slate-600 hover:bg-slate-50 hover:text-slate-900 capitalize transition-colors"
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-3 py-4 px-6 bg-slate-900 hover:bg-black disabled:bg-slate-700 text-white text-[11px] font-black uppercase tracking-[0.25em] rounded-2xl transition-all active:scale-[0.98] shadow-xl shadow-slate-200"
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : (
            <>
              {mode === 'login' ? 'Login' : 'Signup'}
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </form>

      <div className="mt-10 pt-8 border-t border-slate-50 text-center">
        <button
          onClick={onToggle}
          className="text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-slate-900 transition-colors focus:outline-none"
        >
          {mode === 'login' ? "Need an account?" : "Already have an account?"}
        </button>
      </div>
    </div>
  );
};
