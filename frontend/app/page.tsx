'use client';

import axios from 'axios';
import { useState } from 'react';
import InpaintingCanvas from '@/components/InpaintingCanvas';
import { Download, Sparkles, AlertCircle } from 'lucide-react';

export default function Home() {
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (image: Blob, mask: Blob, prompt: string) => {
    setIsGenerating(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', image, 'image.png');
    formData.append('mask', mask, 'mask.png');
    formData.append('prompt', prompt);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/api/generate`, formData, {
        responseType: 'blob',
      });

      const imageUrl = URL.createObjectURL(response.data);
      setGeneratedImage(imageUrl);

      // Scroll to result
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    } catch (err: any) {
      console.error("Error generating image:", err);

      let errorMessage = "Failed to generate image.";

      if (err.response) {
        // Server responded with a status code outside 2xx
        errorMessage = `Server Error (${err.response.status}): ${err.response.data?.detail || err.response.statusText}`;
      } else if (err.request) {
        // Request was made but no response received (CORS or Network Error)
        errorMessage = "Network Error: No response received from backend. Check URL and CORS.";
      } else {
        // Something else happened
        errorMessage = `Error: ${err.message}`;
      }

      setError(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white flex flex-col items-center py-12 px-4 selection:bg-purple-500 selection:text-white">
      {/* Header */}
      <div className="text-center mb-10 max-w-2xl">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="bg-gradient-to-tr from-blue-500 to-purple-600 p-3 rounded-2xl shadow-lg shadow-purple-900/20">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
        </div>
        <h1 className="text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-4 tracking-tight">
          Pencil-to-Inpaint
        </h1>
        <p className="text-gray-400 text-lg leading-relaxed">
          Draw over the area you want to change, describe your imagination, and watch AI bring it to life.
        </p>
      </div>

      {/* Main App */}
      <InpaintingCanvas onGenerate={handleGenerate} isGenerating={isGenerating} />

      {/* Error Message */}
      {error && (
        <div className="mt-8 bg-red-900/20 border border-red-800 text-red-300 px-6 py-4 rounded-xl flex items-center gap-3 max-w-lg">
          <AlertCircle />
          <p>{error}</p>
        </div>
      )}

      {/* Result Section */}
      {generatedImage && (
        <div className="mt-16 bg-gray-900/50 p-8 rounded-2xl border border-gray-800 flex flex-col items-center animate-in fade-in slide-in-from-bottom-8 duration-700">
          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-500 mb-6">Generated Masterpiece</h2>
          <div className="relative group rounded-xl overflow-hidden shadow-2xl shadow-black/50 border border-gray-800">
            <img src={generatedImage} alt="Generated" className="max-w-full max-h-[800px] object-contain" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-center p-6">
              <a
                href={generatedImage}
                download="inpainted-masterpiece.png"
                className="bg-white text-black font-bold px-6 py-3 rounded-full flex items-center gap-2 hover:scale-105 transition-transform"
              >
                <Download size={20} /> Download Image
              </a>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
