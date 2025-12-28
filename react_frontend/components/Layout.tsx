
import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen relative flex flex-col items-center justify-center p-6 bg-[#fcfcfc] selection:bg-slate-200 selection:text-slate-900 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-slate-100/40 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-slate-100/50 rounded-full blur-[120px]" />
        <svg className="absolute inset-0 w-full h-full opacity-[0.03]" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative z-10 w-full flex flex-col items-center gap-10">
        <header className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-slate-900 text-white mb-4 shadow-2xl shadow-slate-200 animate-in zoom-in duration-700">
             <span className="font-black text-xl italic">E</span>
          </div>
          <h1 className="text-4xl font-black tracking-tighter text-slate-900 sm:text-5xl">
            Edu<span className="text-slate-400">Rag</span>
          </h1>
          <p className="text-slate-400 text-[10px] font-black tracking-[0.3em] uppercase opacity-70">
            Enterprise RAG Platform
          </p>
        </header>

        {children}

        <footer className="mt-8 flex items-center gap-8 text-[9px] uppercase tracking-[0.25em] font-black text-slate-300">
          <a href="#" className="hover:text-slate-900 transition-colors">Privacy</a>
          <span className="w-1 h-1 rounded-full bg-slate-200" />
          <a href="#" className="hover:text-slate-900 transition-colors">Security</a>
          <span className="w-1 h-1 rounded-full bg-slate-200" />
          <a href="#" className="hover:text-slate-900 transition-colors">Help</a>
        </footer>
      </div>
    </div>
  );
};
