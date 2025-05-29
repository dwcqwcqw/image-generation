'use client'

import { useState } from 'react'
import TextToImagePanel from '@/components/TextToImagePanel'
import ImageToImagePanel from '@/components/ImageToImagePanel'
import BaseModelSelector from '@/components/BaseModelSelector'
import Header from '@/components/Header'
import { ImageIcon, Type } from 'lucide-react'
import { BaseModelProvider, useBaseModel } from '@/contexts/BaseModelContext'

// Content component that uses the context
function HomePageContent() {
  const [activeTab, setActiveTab] = useState<'text-to-image' | 'image-to-image'>('text-to-image')
  const { baseModel, setBaseModel } = useBaseModel()

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

          {/* Global Base Model Selector */}
          <div className="mb-6">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Style</h2>
              <BaseModelSelector
                value={baseModel}
                onChange={setBaseModel} // Directly use setBaseModel from context
                disabled={false}
              />
            </div>
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

// Main HomePage component that wraps content with Provider
export default function HomePage() {
  return (
    <BaseModelProvider>
      <HomePageContent />
    </BaseModelProvider>
  )
} 