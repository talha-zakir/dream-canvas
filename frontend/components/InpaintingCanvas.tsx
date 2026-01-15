'use client';

import React, { useEffect, useRef, useState } from 'react';
import { fabric } from 'fabric';
import { Upload, Eraser, Pen, Loader2 } from 'lucide-react';

interface InpaintingCanvasProps {
    onGenerate: (image: Blob, mask: Blob, prompt: string) => void;
    isGenerating: boolean;
}

export default function InpaintingCanvas({ onGenerate, isGenerating }: InpaintingCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [fabricCanvas, setFabricCanvas] = useState<fabric.Canvas | null>(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [brushSize, setBrushSize] = useState(20);
    const [prompt, setPrompt] = useState("");
    const [baseImage, setBaseImage] = useState<fabric.Image | null>(null);

    useEffect(() => {
        if (canvasRef.current && !fabricCanvas) {
            const canvas = new fabric.Canvas(canvasRef.current, {
                isDrawingMode: false,
                width: 800,
                height: 600,
                backgroundColor: '#1a1a1a',
            });

            // Configure Brush
            canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
            canvas.freeDrawingBrush.color = 'white';
            canvas.freeDrawingBrush.width = brushSize;

            setFabricCanvas(canvas);

            return () => {
                canvas.dispose();
            }
        }
    }, [canvasRef]);

    useEffect(() => {
        if (fabricCanvas) {
            fabricCanvas.freeDrawingBrush.width = brushSize;
        }
    }, [brushSize, fabricCanvas]);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file || !fabricCanvas) return;

        const reader = new FileReader();
        reader.onload = (f) => {
            const data = f.target?.result as string;
            fabric.Image.fromURL(data, (img) => {
                // Clear previous
                fabricCanvas.clear();
                fabricCanvas.setBackgroundColor('#1a1a1a', () => { });

                // Scale image to fit within 1024x1024 or canvas size
                const maxSize = 800;
                let scale = 1;
                if (img.width! > maxSize || img.height! > maxSize) {
                    scale = Math.min(maxSize / img.width!, maxSize / img.height!);
                }

                img.scale(scale);
                img.set({
                    left: (fabricCanvas.width! - img.width! * scale) / 2,
                    top: (fabricCanvas.height! - img.height! * scale) / 2,
                    selectable: false,
                    evented: false,
                });

                fabricCanvas.add(img);
                setBaseImage(img);
                fabricCanvas.isDrawingMode = true; // Enable drawing immediately
                setImageLoaded(true);
                fabricCanvas.renderAll();
            });
        };
        reader.readAsDataURL(file);
    };

    const handleClearMask = () => {
        if (!fabricCanvas || !baseImage) return;
        fabricCanvas.getObjects().forEach((obj) => {
            if (obj !== baseImage) {
                fabricCanvas.remove(obj);
            }
        });
    };

    const generate = async () => {
        if (!fabricCanvas || !baseImage) return;

        // 1. Get the original image (cropping to the actual image area)
        // We need to export the area where the image is.

        // Create a temporary canvas to extract the mask
        // OR, we can just export the whole canvas if we enforce the image fills it, 
        // but here we centered the image.

        // Simpler approach: Export the whole canvas as the mask, 
        // but we need to make sure the background is BLACK and drawing is WHITE.
        // The base image should be invisible for the mask.

        const originalVisibility = baseImage.visible;

        // A. Generate MASK
        baseImage.visible = false;
        fabricCanvas.setBackgroundColor('black', () => { });
        const maskDataURL = fabricCanvas.toDataURL({
            format: 'png',
            quality: 1,
        });

        // B. Generate ORIGINAL
        baseImage.visible = true;
        // Hide drawings
        const drawings = fabricCanvas.getObjects().filter(o => o !== baseImage);
        drawings.forEach(d => d.visible = false);
        fabricCanvas.setBackgroundColor('black', () => { }); // Or whatever

        const imageDataURL = fabricCanvas.toDataURL({
            format: 'png',
            quality: 1,
        });

        // Reset visibility
        drawings.forEach(d => d.visible = true);
        baseImage.visible = originalVisibility;
        fabricCanvas.renderAll();

        // Convert DataURLs to Blobs
        const maskBlob = await (await fetch(maskDataURL)).blob();
        const imageBlob = await (await fetch(imageDataURL)).blob();

        onGenerate(imageBlob, maskBlob, prompt);
    };

    return (
        <div className="flex flex-col gap-4 w-full max-w-4xl mx-auto p-4">
            {/* Controls */}
            <div className="flex flex-wrap gap-4 items-center bg-gray-900 p-4 rounded-xl border border-gray-800">
                <div className="relative">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                        id="file-upload"
                    />
                    <label
                        htmlFor="file-upload"
                        className="flex items-center gap-2 cursor-pointer bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white font-medium transition-colors"
                    >
                        <Upload size={18} /> Upload Image
                    </label>
                </div>

                <div className="h-8 w-[1px] bg-gray-700 mx-2"></div>

                <div className="flex items-center gap-2 text-white">
                    <Pen size={18} className="text-gray-400" />
                    <input
                        type="range"
                        min="5"
                        max="100"
                        value={brushSize}
                        onChange={(e) => setBrushSize(Number(e.target.value))}
                        className="w-32 accent-blue-500"
                    />
                </div>

                <button
                    onClick={handleClearMask}
                    className="flex items-center gap-2 text-gray-300 hover:text-white px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors"
                >
                    <Eraser size={18} /> Clear Mask
                </button>
            </div>

            {/* Canvas Area */}
            <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-[#1a1a1a] shadow-2xl">
                <canvas ref={canvasRef} className="w-full" />

                {!imageLoaded && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 pointer-events-none">
                        <Upload size={48} className="mb-4 opacity-50" />
                        <p>Upload an image to start inpainting</p>
                    </div>
                )}
            </div>

            {/* Prompt & Generate */}
            <div className="flex gap-2">
                <input
                    type="text"
                    placeholder="Describe what to fill in the masked area..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="flex-1 bg-gray-900 border border-gray-800 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-500"
                    disabled={!imageLoaded || isGenerating}
                />
                <button
                    onClick={generate}
                    disabled={!imageLoaded || !prompt || isGenerating}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold px-8 py-3 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
                >
                    {isGenerating && <Loader2 className="animate-spin" />}
                    {isGenerating ? 'Dreaming...' : 'Generate'}
                </button>
            </div>
        </div>
    );
}
