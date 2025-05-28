'use client'

import { useState } from 'react'
import TextToImagePanel from '@/components/TextToImagePanel'
import ImageToImagePanel from '@/components/ImageToImagePanel'
import Header from '@/components/Header'
import { ImageIcon, Type } from 'lucide-react'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'text-to-image' | 'image-to-image'>('text-to-image')

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              AI Image Generation
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Create stunning images with FLUX AI model. Generate from text prompts or transform existing images.
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
            <div className="border-b border-gray-200 bg-gray-50">
              <div className="flex">
                <button
                  onClick={() => setActiveTab('text-to-image')}
                  className={`flex-1 px-6 py-4 text-sm font-medium text-center border-b-2 transition-colors duration-200 ${
                    activeTab === 'text-to-image'
                      ? 'border-primary-500 text-primary-600 bg-white'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-2">
                    <Type className="w-4 h-4" />
                    <span>Text to Image</span>
                  </div>
                </button>
                
                <button
                  onClick={() => setActiveTab('image-to-image')}
                  className={`flex-1 px-6 py-4 text-sm font-medium text-center border-b-2 transition-colors duration-200 ${
                    activeTab === 'image-to-image'
                      ? 'border-primary-500 text-primary-600 bg-white'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-2">
                    <ImageIcon className="w-4 h-4" />
                    <span>Image to Image</span>
                  </div>
                </button>
              </div>
            </div>

            <div className="p-6">
              {activeTab === 'text-to-image' && <TextToImagePanel />}
              {activeTab === 'image-to-image' && <ImageToImagePanel />}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
} 