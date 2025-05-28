'use client'

import { useState } from 'react'
import { toast } from 'react-hot-toast'
import { 
  Play, 
  Download, 
  RefreshCw, 
  Settings, 
  ChevronDown,
  ChevronUp,
  Shuffle
} from 'lucide-react'
import ImageGallery from './ImageGallery'
import { generateTextToImage } from '@/services/api'
import type { TextToImageParams, GeneratedImage } from '@/types'

export default function TextToImagePanel() {
  const [isLoading, setIsLoading] = useState(false)
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  
  const [params, setParams] = useState<TextToImageParams>({
    prompt: '',
    negativePrompt: '',
    width: 512,
    height: 512,
    steps: 20,
    cfgScale: 7,
    seed: -1,
    numImages: 1,
  })

  const handleGenerate = async () => {
    if (!params.prompt.trim()) {
      toast.error('Please enter a prompt')
      return
    }

    setIsLoading(true)
    try {
      const result = await generateTextToImage(params)
      setGeneratedImages(prev => [...result, ...prev])
      toast.success(`Generated ${result.length} image(s)`)
    } catch (error) {
      console.error('Generation failed:', error)
      toast.error('Failed to generate image. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleRandomSeed = () => {
    setParams(prev => ({ ...prev, seed: Math.floor(Math.random() * 1000000) }))
  }

  const presetSizes = [
    { label: '512×512', width: 512, height: 512 },
    { label: '768×768', width: 768, height: 768 },
    { label: '1024×1024', width: 1024, height: 1024 },
    { label: '512×768', width: 512, height: 768 },
    { label: '768×512', width: 768, height: 512 },
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Panel */}
        <div className="lg:col-span-1 space-y-4">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Text to Image</h3>
            
            {/* Prompt */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Positive Prompt *
              </label>
              <textarea
                value={params.prompt}
                onChange={(e) => setParams(prev => ({ ...prev, prompt: e.target.value }))}
                placeholder="A beautiful landscape with mountains and lake at sunset..."
                className="textarea-field h-24"
                disabled={isLoading}
              />
            </div>

            {/* Negative Prompt */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Negative Prompt
              </label>
              <textarea
                value={params.negativePrompt}
                onChange={(e) => setParams(prev => ({ ...prev, negativePrompt: e.target.value }))}
                placeholder="blurry, low quality, distorted..."
                className="textarea-field h-20"
                disabled={isLoading}
              />
            </div>

            {/* Size Presets */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Image Size
              </label>
              <div className="grid grid-cols-2 gap-2">
                {presetSizes.map((size) => (
                  <button
                    key={size.label}
                    onClick={() => setParams(prev => ({ 
                      ...prev, 
                      width: size.width, 
                      height: size.height 
                    }))}
                    className={`p-2 text-sm rounded-lg border transition-colors ${
                      params.width === size.width && params.height === size.height
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                    disabled={isLoading}
                  >
                    {size.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Number of Images */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Number of Images
              </label>
              <div className="grid grid-cols-4 gap-2">
                {[1, 2, 3, 4].map((num) => (
                  <button
                    key={num}
                    onClick={() => setParams(prev => ({ ...prev, numImages: num }))}
                    className={`p-2 text-sm rounded-lg border transition-colors ${
                      params.numImages === num
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                    disabled={isLoading}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            {/* Advanced Settings Toggle */}
            <button
              onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
              className="flex items-center justify-between w-full p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center space-x-2">
                <Settings className="w-4 h-4" />
                <span className="font-medium">Advanced Settings</span>
              </div>
              {isAdvancedOpen ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {/* Advanced Settings */}
            {isAdvancedOpen && (
              <div className="space-y-4 pt-2">
                {/* Steps */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Steps: {params.steps}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={params.steps}
                    onChange={(e) => setParams(prev => ({ ...prev, steps: Number(e.target.value) }))}
                    className="slider"
                    disabled={isLoading}
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>10</span>
                    <span>50</span>
                  </div>
                </div>

                {/* CFG Scale */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">
                    CFG Scale: {params.cfgScale}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    value={params.cfgScale}
                    onChange={(e) => setParams(prev => ({ ...prev, cfgScale: Number(e.target.value) }))}
                    className="slider"
                    disabled={isLoading}
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>1</span>
                    <span>20</span>
                  </div>
                </div>

                {/* Seed */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Seed
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      value={params.seed === -1 ? '' : params.seed}
                      onChange={(e) => setParams(prev => ({ 
                        ...prev, 
                        seed: e.target.value === '' ? -1 : Number(e.target.value) 
                      }))}
                      placeholder="Random"
                      className="input-field flex-1"
                      disabled={isLoading}
                    />
                    <button
                      onClick={handleRandomSeed}
                      className="btn-secondary px-3"
                      disabled={isLoading}
                    >
                      <Shuffle className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isLoading || !params.prompt.trim()}
              className="btn-primary w-full py-3 text-base font-semibold"
            >
              {isLoading ? (
                <div className="flex items-center justify-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Generating...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Play className="w-4 h-4" />
                  <span>Generate</span>
                </div>
              )}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          <ImageGallery 
            images={generatedImages}
            isLoading={isLoading}
            title="Generated Images"
          />
        </div>
      </div>
    </div>
  )
} 