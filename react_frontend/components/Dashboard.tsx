import React, { useState, useRef, useEffect } from "react";
import {
  FileText,
  MessageSquare,
  LineChart,
  LogOut,
  Upload,
  Sparkles,
  Send,
  FileCheck,
  User as UserIcon,
  Bot,
  ShieldCheck,
  LayoutGrid,
  Github,
} from "lucide-react";
import { UserProfile, UploadedFile, Role } from "../types";

const API_URL = "http://localhost:8000";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: Date;
}

interface DashboardProps {
  user: UserProfile;
  onLogout: () => void;
}

type Tab = "documents" | "chat" | "insights";

// Helper function to format the AI response text
const formatMessageContent = (text: string) => {
  // 1. Ensure text preceding a bullet point (*) has a newline
  let formatted = text.replace(/([^\n])\s*\*\s/g, "$1\n\n• ");

  // 2. Replace starting * with a bullet point
  formatted = formatted.replace(/^\*\s/g, "• ");

  // 3. Clean up multiple newlines if any
  return formatted.trim();
};

export const Dashboard: React.FC<DashboardProps> = ({ user, onLogout }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("documents");
  const [uploadProgress, setUploadProgress] = useState(0);

  // State initialization from LocalStorage
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>(() => {
    const saved = localStorage.getItem("uploadedFiles");
    return saved ? JSON.parse(saved) : [];
  });

  const [selectedFileId, setSelectedFileId] = useState<string | null>(() => {
    return localStorage.getItem("selectedFileId") || null;
  });

  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionResult, setExtractionResult] = useState<string | null>(null);
  const [selectedFileToUpload, setSelectedFileToUpload] = useState<File | null>(
    null
  );
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem("uploadedFiles", JSON.stringify(uploadedFiles));
  }, [uploadedFiles]);

  useEffect(() => {
    if (selectedFileId) {
      localStorage.setItem("selectedFileId", selectedFileId);
    }
  }, [selectedFileId]);

  const getAuthHeader = () => {
    const password = (user as any).password || "";
    return `Basic ${btoa(`${user.username}:${password}`)}`;
  };

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFileToUpload(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFileToUpload) return;

    // Reset progress and start simulation
    setUploadProgress(0);

    // Create a simulated progress interval
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) return prev; // Hold at 90% until actually done
        return prev + 10;
      });
    }, 500);

    try {
      const formData = new FormData();
      formData.append("file", selectedFileToUpload);

      const response = await fetch(`${API_URL}/docs/upload_docs`, {
        method: "POST",
        headers: {
          Authorization: getAuthHeader(),
        },
        body: formData,
      });

      // Clear interval and complete progress
      clearInterval(interval);
      setUploadProgress(100);

      if (response.ok) {
        const newFile: UploadedFile = {
          id: Math.random().toString(36).substr(2, 9),
          name: selectedFileToUpload.name,
          owner: user.username,
          visibility: "admin",
          timestamp: new Date().toLocaleDateString(),
        };

        // Small delay to let the user see the 100% bar before resetting
        setTimeout(() => {
          setUploadedFiles([newFile, ...uploadedFiles]);
          setSelectedFileToUpload(null);
          if (fileInputRef.current) fileInputRef.current.value = "";
          setSelectedFileId(newFile.id);
          setUploadProgress(0); // Reset progress
        }, 800);
      } else {
        alert("Document upload failed");
        setUploadProgress(0);
      }
    } catch (error) {
      clearInterval(interval);
      setUploadProgress(0);
      console.error("Upload error:", error);
      alert("Failed to connect to server");
    }
  };

  const selectedFile = uploadedFiles.find((f) => f.id === selectedFileId);

  const handleSendMessage = async () => {
    if (!inputText.trim() || !selectedFileId || !selectedFile) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: "user",
      timestamp: new Date(),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setInputText("");

    try {
      const contextAwarePrompt = `From document "${selectedFile.name}" provide me the info: ${userMsg.text}`;

      const formData = new URLSearchParams();
      formData.append("message", contextAwarePrompt);

      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          Authorization: getAuthHeader(),
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const aiMsg: Message = {
          id: (Date.now() + 1).toString(),
          text: data.answer || "No response received.",
          sender: "ai",
          timestamp: new Date(),
        };
        setChatMessages((prev) => [...prev, aiMsg]);
      } else {
        const errorMsg: Message = {
          id: (Date.now() + 1).toString(),
          text: "Failed to retrieve answer from server.",
          sender: "ai",
          timestamp: new Date(),
        };
        setChatMessages((prev) => [...prev, errorMsg]);
      }
    } catch (error) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: "Error connecting to server.",
        sender: "ai",
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, errorMsg]);
    }
  };

  const handleExtraction = async (type: string) => {
    if (!selectedFileId || !selectedFile) return;
    setIsExtracting(true);

    try {
      const baseAction =
        type === "Summary"
          ? "summarize the document"
          : "extract important information";
      const contextAwareMessage = `From document "${selectedFile.name}", please ${baseAction}`;

      const formData = new URLSearchParams();
      formData.append("message", contextAwareMessage);

      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          Authorization: getAuthHeader(),
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setExtractionResult(data.answer);
      } else {
        setExtractionResult("Failed to extract information.");
      }
    } catch (error) {
      setExtractionResult("Error connecting to server.");
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#fafafa] text-[#0f172a] overflow-hidden font-sans">
      <aside className="w-72 border-r border-slate-200 bg-white flex flex-col shrink-0 z-30 h-full">
        <div className="p-8 pb-6">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center text-white text-xs font-black italic">
              E
            </div>
            <h1 className="text-xl font-black tracking-tighter">
              Edu<span className="text-slate-400 font-bold">Rag</span>
            </h1>
          </div>
          <p className="text-[10px] font-bold text-slate-500 mb-6">
            Enterprise RAG Platform
          </p>
          <p className="text-[9px] uppercase tracking-[0.25em] font-black text-slate-400">
            Main Menu
          </p>
        </div>

        <div className="px-6 mb-6">
          <div className="flex items-center gap-4 p-4 bg-slate-50 border border-slate-100 rounded-[1.25rem]">
            <div className="w-10 h-10 rounded-full bg-slate-900 flex items-center justify-center text-white shadow-lg shadow-slate-200 font-black text-sm uppercase">
              {user.username.slice(0, 2)}
            </div>
            <div className="overflow-hidden">
              <p className="text-[13px] font-bold text-slate-900 truncate">
                {user.username}
              </p>
              <div className="flex items-center gap-1">
                <ShieldCheck size={10} className="text-slate-900" />
                <span className="text-[9px] uppercase tracking-wider font-black text-slate-400">
                  {user.role}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="px-4 space-y-1 shrink-0">
          <SidebarItem
            icon={<LayoutGrid size={18} />}
            label="Documents"
            active={activeTab === "documents"}
            onClick={() => setActiveTab("documents")}
          />
          <SidebarItem
            icon={<MessageSquare size={18} />}
            label="Chat"
            active={activeTab === "chat"}
            onClick={() => setActiveTab("chat")}
          />
          <SidebarItem
            icon={<LineChart size={18} />}
            label="Insights"
            active={activeTab === "insights"}
            onClick={() => setActiveTab("insights")}
          />
        </nav>

        {/* NEW SECTION: File List in Sidebar */}
        <div className="flex-1 overflow-y-auto px-4 mt-8 mb-4 min-h-0">
          <p className="px-2 text-[9px] uppercase tracking-[0.25em] font-black text-slate-400 mb-4 sticky top-0 bg-white z-10 py-2">
            Recent Files
          </p>
          <div className="space-y-1">
            {uploadedFiles.length === 0 ? (
              <p className="text-xs text-slate-300 font-medium px-2 italic">
                No files yet
              </p>
            ) : (
              uploadedFiles.map((file) => (
                <button
                  key={file.id}
                  onClick={() => setSelectedFileId(file.id)}
                  className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl text-xs font-bold transition-all duration-200 group text-left ${
                    selectedFileId === file.id
                      ? "bg-slate-100 text-slate-900 border border-slate-200"
                      : "text-slate-500 hover:text-slate-900 hover:bg-slate-50 border border-transparent"
                  }`}
                >
                  <FileText
                    size={14}
                    className={
                      selectedFileId === file.id
                        ? "text-slate-900"
                        : "text-slate-400 group-hover:text-slate-600"
                    }
                  />
                  <span className="truncate">{file.name}</span>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Footer Actions: GitHub & Logout */}
        <div className="p-4 border-t border-slate-100 shrink-0 mt-auto space-y-2">
          {/* GitHub Repo Link */}
          <a
            href="https://github.com/Ismail007-Sk/Intel-Unnati-Industrial-Training_PS-4" // <--- PUT YOUR LINK HERE
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center gap-3 px-5 py-4 text-slate-400 hover:text-slate-900 hover:bg-slate-50 rounded-2xl transition-all group"
          >
            <Github
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="text-[10px] font-black uppercase tracking-[0.2em]">
              Source Code
            </span>
          </a>

          {/* Logout Button */}
          <button
            onClick={onLogout}
            className="w-full flex items-center gap-3 px-5 py-4 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-2xl transition-all group"
          >
            <LogOut
              size={18}
              className="group-hover:-translate-x-1 transition-transform"
            />
            <span className="text-[10px] font-black uppercase tracking-[0.2em]">
              Logout
            </span>
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-20 border-b border-slate-100 bg-white/50 backdrop-blur-xl px-10 flex items-center justify-between shrink-0 sticky top-0 z-20">
          <div className="flex items-center gap-6">
            <h2 className="text-sm font-black tracking-[0.2em] text-slate-900 uppercase">
              {activeTab}
            </h2>
          </div>
        </header>

        <div
          className={`flex-1 overflow-y-auto ${
            activeTab === "chat" ? "px-0 pt-0" : "p-10"
          } max-w-7xl mx-auto w-full flex flex-col`}
        >
          {activeTab === "documents" && (
            <div className="space-y-12 animate-in fade-in duration-700">
              <section className="bg-white p-10 rounded-[2.5rem] border border-slate-100 shadow-sm space-y-8">
                <div className="flex items-center justify-between">
                  <h5 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                    Upload Document
                  </h5>
                </div>
                <div className="flex flex-col md:flex-row items-end gap-4">
                  <div className="flex-1 space-y-2">
                    <label className="text-[9px] uppercase font-black text-slate-300 ml-1">
                      Choose PDF
                    </label>
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className={`h-14 px-5 bg-slate-50 border rounded-2xl flex items-center gap-3 text-sm transition-all cursor-pointer group ${
                        selectedFileToUpload
                          ? "border-slate-300 bg-slate-100"
                          : "border-slate-100 hover:border-slate-300"
                      }`}
                    >
                      <div
                        className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          selectedFileToUpload
                            ? "bg-slate-900 text-white"
                            : "bg-white border border-slate-100 text-slate-300"
                        }`}
                      >
                        <FileText size={16} />
                      </div>
                      <span
                        className={
                          selectedFileToUpload
                            ? "text-slate-900 font-bold"
                            : "text-slate-400 font-medium italic"
                        }
                      >
                        {selectedFileToUpload
                          ? selectedFileToUpload.name
                          : "Select a file..."}
                      </span>
                      <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept=".pdf"
                        onChange={handleFileChange}
                        title="Select a PDF file to upload"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleUpload}
                    disabled={!selectedFileToUpload || uploadProgress > 0}
                    className="relative h-14 w-40 overflow-hidden bg-slate-900 text-white rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] flex items-center justify-center gap-3 hover:bg-black transition-all active:scale-95 shadow-xl shadow-slate-200 disabled:opacity-70 disabled:active:scale-100"
                  >
                    {/* Progress Bar Background */}
                    {uploadProgress > 0 && (
                      <div
                        className="absolute left-0 top-0 bottom-0 bg-slate-700 transition-all duration-300 ease-out"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    )}

                    {/* Button Content (z-10 to sit on top of progress bar) */}
                    <div className="relative z-10 flex items-center gap-3">
                      {uploadProgress > 0 ? (
                        <>
                          {/* Simple Spinner */}
                          <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          {uploadProgress === 100 ? "Done" : "Uploading..."}
                        </>
                      ) : (
                        <>
                          <Upload size={16} /> Upload
                        </>
                      )}
                    </div>
                  </button>
                </div>
              </section>

              <section className="space-y-6">
                <div className="flex items-center justify-between px-2">
                  <h5 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                    All Files
                  </h5>
                  <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">
                    {uploadedFiles.length} uploaded
                  </span>
                </div>

                {uploadedFiles.length === 0 ? (
                  <div className="p-24 text-center bg-white rounded-[3rem] border border-slate-100">
                    <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
                      <FileCheck className="text-slate-200" size={32} />
                    </div>
                    <p className="text-slate-400 font-medium italic">
                      No documents available.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {uploadedFiles.map((file) => (
                      <div
                        key={file.id}
                        onClick={() => setSelectedFileId(file.id)}
                        className={`px-6 py-4 bg-white border rounded-xl transition-all cursor-pointer flex items-center justify-between group hover:bg-slate-50 ${
                          selectedFileId === file.id
                            ? "border-slate-900 bg-slate-50 shadow-md"
                            : "border-slate-100 shadow-sm"
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <FileText
                            size={18}
                            className={
                              selectedFileId === file.id
                                ? "text-slate-900"
                                : "text-slate-300"
                            }
                          />
                          <p
                            className={`text-sm font-medium ${
                              selectedFileId === file.id
                                ? "text-slate-900 font-bold"
                                : "text-slate-700"
                            }`}
                          >
                            {file.name} —{" "}
                            <span className="opacity-60 capitalize font-normal">
                              {file.visibility}
                            </span>
                          </p>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-[9px] font-black uppercase tracking-widest text-slate-300">
                            {file.timestamp}
                          </span>
                          {selectedFileId === file.id && (
                            <div className="w-2 h-2 rounded-full bg-slate-900" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}

          {activeTab === "chat" && (
            <div className="flex-1 flex flex-col h-full bg-white relative animate-in fade-in duration-500">
              <div className="flex-1 overflow-y-auto px-10 pt-16 pb-48 flex flex-col">
                <div className="max-w-4xl mx-auto w-full space-y-12">
                  <div className="mb-12 space-y-6">
                    <div className="space-y-2">
                      <h4 className="text-4xl font-black text-slate-900 tracking-tighter leading-none">
                        Hello{" "}
                        <span className="text-slate-900">{user.username}</span>,
                      </h4>
                      <p className="text-xl font-medium text-slate-400 tracking-tight">
                        How can I help you today?
                      </p>
                    </div>

                    {selectedFile && (
                      <div className="inline-flex items-center gap-3 px-5 py-2.5 bg-slate-50 rounded-2xl border border-slate-100">
                        <FileText size={16} className="text-slate-900" />
                        <span className="text-xs font-black text-slate-900 uppercase tracking-widest">
                          Context: {selectedFile.name}
                        </span>
                      </div>
                    )}
                  </div>

                  {!selectedFile ? (
                    <div className="py-24 text-center bg-slate-50 rounded-[3rem] border border-slate-200 border-dashed animate-in zoom-in duration-500">
                      <div className="w-20 h-20 bg-white rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-sm border border-slate-100">
                        <MessageSquare size={32} className="text-slate-100" />
                      </div>
                      <p className="text-slate-400 font-bold uppercase tracking-widest text-[11px]">
                        Select a document to begin.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-10">
                      {chatMessages.map((msg) => (
                        <div
                          key={msg.id}
                          className={`flex ${
                            msg.sender === "user"
                              ? "justify-end"
                              : "justify-start"
                          } animate-in slide-in-from-bottom-4 duration-500`}
                        >
                          <div
                            className={`flex gap-6 max-w-[90%] ${
                              msg.sender === "user"
                                ? "flex-row-reverse"
                                : "flex-row"
                            }`}
                          >
                            <div
                              className={`w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 shadow-sm border ${
                                msg.sender === "user"
                                  ? "bg-slate-900 border-slate-900"
                                  : "bg-white border border-slate-100 text-slate-900"
                              }`}
                            >
                              {msg.sender === "user" ? (
                                <UserIcon size={18} className="text-white" />
                              ) : (
                                <Bot size={22} />
                              )}
                            </div>

                            {/* CHANGED: Added whitespace-pre-wrap to respect newlines and formatted content */}
                            <div
                              className={`p-6 rounded-[2.25rem] text-[16px] leading-relaxed shadow-sm whitespace-pre-wrap ${
                                msg.sender === "user"
                                  ? "bg-slate-100 border border-slate-200 rounded-tr-none text-slate-900"
                                  : "bg-white border border-slate-100 rounded-tl-none text-slate-700"
                              }`}
                            >
                              {/* Using the formatter only for AI messages to keep user text raw */}
                              {msg.sender === "ai"
                                ? formatMessageContent(msg.text)
                                : msg.text}
                            </div>
                          </div>
                        </div>
                      ))}
                      <div ref={chatEndRef} />
                    </div>
                  )}
                </div>
              </div>

              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-12 pb-10 px-10 pointer-events-none">
                <div className="max-w-3xl mx-auto w-full pointer-events-auto">
                  <div className="bg-white border border-slate-200 rounded-3xl flex flex-col p-4 transition-all relative">
                    <div className="flex justify-end pr-2 pt-2 absolute top-0 right-0 z-10">
                      <button
                        onClick={handleSendMessage}
                        disabled={!selectedFile || !inputText.trim()}
                        className="bg-slate-900 text-white p-3 rounded-2xl hover:bg-black transition-all active:scale-95 disabled:opacity-10 shadow-lg shadow-slate-200 focus:outline-none"
                        title="Send Message"
                      >
                        <Send size={18} />
                      </button>
                    </div>
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      disabled={!selectedFile}
                      placeholder="Ask a question..."
                      className="bg-transparent border-none focus:ring-0 focus:outline-none text-[16px] p-4 pr-16 resize-none min-h-[100px] w-full placeholder:text-slate-400 text-slate-700 font-medium outline-none shadow-none ring-0 appearance-none"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "insights" && (
            <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
              <section className="bg-white p-12 rounded-[3rem] border border-slate-100 shadow-sm space-y-10">
                <div className="space-y-2">
                  <h4 className="text-2xl font-black text-slate-900 tracking-tighter">
                    Extract Insights
                  </h4>
                  <p className="text-sm text-slate-400 font-medium">
                    Analyze content from:{" "}
                    <span className="text-slate-900 font-bold">
                      {selectedFile?.name || "No file selected"}
                    </span>
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ActionButton
                    label="Extract Important Information"
                    disabled={!selectedFile}
                    onClick={() => handleExtraction("Important Information")}
                  />
                  <ActionButton
                    label="Extract Summary"
                    disabled={!selectedFile}
                    onClick={() => handleExtraction("Summary")}
                  />
                </div>

                {isExtracting ? (
                  <div className="p-20 border border-slate-100 rounded-[2.5rem] flex flex-col items-center justify-center text-center bg-slate-50/50 animate-pulse">
                    <div className="w-16 h-16 rounded-full border-[6px] border-slate-100 border-t-slate-900 animate-spin mb-6 shadow-inner"></div>
                    <p className="text-slate-400 text-[11px] font-black uppercase tracking-[0.3em]">
                      Processing...
                    </p>
                  </div>
                ) : extractionResult ? (
                  <div className="p-10 bg-slate-900 rounded-[2.5rem] shadow-2xl animate-in slide-in-from-top-6 relative overflow-hidden">
                    <div className="relative z-10">
                      <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-3">
                          <h6 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-300">
                            Analysis Results
                          </h6>
                        </div>
                        <button
                          onClick={() => setExtractionResult(null)}
                          className="text-slate-500 hover:text-white transition-colors text-[10px] font-black uppercase tracking-widest"
                        >
                          Clear
                        </button>
                      </div>

                      {/* CHANGED: Added whitespace-pre-wrap here too for consistency */}
                      <div className="text-lg text-slate-200 whitespace-pre-wrap leading-relaxed font-medium">
                        {formatMessageContent(extractionResult)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-24 border-2 border-dashed border-slate-100 rounded-[3rem] flex flex-col items-center justify-center text-center">
                    <Sparkles size={48} className="text-slate-50 mb-6" />
                    <p className="text-slate-400 font-medium italic">
                      Select a file to run analysis.
                    </p>
                  </div>
                )}
              </section>
            </div>
          )}
        </div>
        <footer className="mt-8 flex items-center justify-center gap-8 text-[9px] uppercase tracking-[0.25em] font-black text-slate-300 pb-4 shrink-0">
          <span className="hover:text-slate-900 transition-colors cursor-default">
            Made by Supratim, Ismail, Sanchari
          </span>
        </footer>
      </main>
    </div>
  );
};

const SidebarItem: React.FC<{
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
}> = ({ icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-4 px-5 py-4 rounded-2xl text-sm font-bold tracking-tight transition-all duration-300 ${
      active
        ? "bg-slate-900 text-white shadow-2xl shadow-slate-200 translate-x-1"
        : "text-slate-400 hover:text-slate-900 hover:bg-slate-50"
    }`}
  >
    <span className={active ? "text-white" : "text-slate-300"}>{icon}</span>
    <span className="uppercase tracking-[0.15em] text-[10px] font-black">
      {label}
    </span>
  </button>
);

const ActionButton: React.FC<{
  label: string;
  disabled?: boolean;
  onClick?: () => void;
}> = ({ label, disabled, onClick }) => (
  <button
    disabled={disabled}
    onClick={onClick}
    className="flex items-center justify-center w-full p-10 bg-white border border-slate-100 rounded-[2.5rem] text-[11px] font-black uppercase tracking-[0.25em] text-slate-700 hover:border-slate-900 hover:shadow-2xl hover:-translate-y-1.5 transition-all active:scale-[0.99] group text-center shadow-sm disabled:opacity-10"
  >
    <span>{label}</span>
  </button>
);
