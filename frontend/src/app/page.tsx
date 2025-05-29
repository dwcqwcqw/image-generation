'use client'

import { useState, createContext, useContext } from 'react'
import TextToImagePanel from '@/components/TextToImagePanel'
import ImageToImagePanel from '@/components/ImageToImagePanel'
import BaseModelSelector from '@/components/BaseModelSelector'
import Header from '@/components/Header'
import { ImageIcon, Type } from 'lucide-react'
import type { BaseModelType, LoRAConfig } from '@/types'

// Create context for base model state
interface BaseModelContextType {
  baseModel: BaseModelType
  setBaseModel: (model: BaseModelType) => void
  loraConfig: LoRAConfig
  setLoraConfig: (config: LoRAConfig) => void
}

const BaseModelContext = createContext<BaseModelContextType | undefined>(undefined)

export const useBaseModel = () => {
  const context = useContext(BaseModelContext)
  if (!context) {
    throw new Error('useBaseModel must be used within a BaseModelProvider')
  }
  return context
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'text-to-image' | 'image-to-image'>('text-to-image')
  const [baseModel, setBaseModel] = useState<BaseModelType>('realistic')
  const [loraConfig, setLoraConfig] = useState<LoRAConfig>({ flux_nsfw: 1.0 })

  const handleBaseModelChange = (newBaseModel: BaseModelType) => {
    setBaseModel(newBaseModel)
    // Auto-configure corresponding LoRA
    const newLoRAConfig: LoRAConfig = newBaseModel === 'realistic' 
      ? { flux_nsfw: 1.0 } 
      : { gayporn: 1.0 }
    setLoraConfig(newLoRAConfig)
  }

  return (
    <BaseModelContext.Provider value={{
      baseModel,
      setBaseModel,
      loraConfig,
      setLoraConfig
    }}>
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
                  onChange={handleBaseModelChange}
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
    </BaseModelContext.Provider>
  )
} 