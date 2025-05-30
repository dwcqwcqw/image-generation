'use client'

import React, { useState, useRef } from 'react'
import { toast } from 'react-hot-toast'
import { 
  Play, 
  Download, 
  RefreshCw, 
  Settings, 
  ChevronDown,
  ChevronUp,
  Shuffle,
  StopCircle,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react'
import ImageGallery from './ImageGallery'
import LoRASelector from './LoRASelector'
import { useBaseModel } from '@/contexts/BaseModelContext'
import { generateTextToImage } from '@/services/api'
import { downloadAllCloudflareImages } from '@/utils/cloudflareImageProxy'
import type { TextToImageParams, GeneratedImage } from '@/types'

type GenerationStatus = 'idle' | 'pending' | 'success' | 'error' | 'cancelled'

export default function TextToImagePanel() {
  const [status, setStatus] = useState<GenerationStatus>('idle')
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(true)
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  const [currentGenerationImages, setCurrentGenerationImages] = useState<GeneratedImage[]>([])
  const [historyImages, setHistoryImages] = useState<GeneratedImage[]>([])
  const [currentError, setCurrentError] = useState<string | null>(null)
  const [generationProgress, setGenerationProgress] = useState<string>('')
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // Use global base model state
  const { baseModel, loraConfig, setLoraConfig } = useBaseModel()
  
  const [params, setParams] = useState<TextToImageParams>({
    prompt: '',
    negativePrompt: '', // Will be removed from UI
    width: 512,
    height: 512,
    steps: baseModel === 'realistic' ? 12 : 20, // FLUX uses 12 steps, anime uses 20
    cfgScale: baseModel === 'realistic' ? 1.0 : 7.0, // FLUX uses 1.0, anime uses 7.0
    seed: -1,
    numImages: 1,
    baseModel: baseModel,
    lora_config: loraConfig,
  })

  // Update params when global base model changes
  React.useEffect(() => {
    console.log('BaseModel changed to:', baseModel)
    console.log('LoRA config changed to:', loraConfig)
    
    setParams(prev => ({
      ...prev,
      baseModel: baseModel,
      lora_config: loraConfig,
      // Adjust default parameters based on model type
      steps: baseModel === 'realistic' ? 12 : 20,
      cfgScale: baseModel === 'realistic' ? 1.0 : 7.0,
    }))
  }, [baseModel, loraConfig])

  const handleGenerate = async () => {
    if (!params.prompt.trim()) {
      toast.error('Please enter a prompt')
      return
    }

    setStatus('pending')
    setCurrentError(null)
    setGenerationProgress('Preparing generation...')
    setCurrentGenerationImages([])
    
    // Create new AbortController for this generation
    abortControllerRef.current = new AbortController()
    
    try {
      console.log('Generating with params:', params)
      
      const result = await generateTextToImage(params, abortControllerRef.current.signal)
      
      if (abortControllerRef.current.signal.aborted) {
        setStatus('cancelled')
        setGenerationProgress('Generation cancelled')
        return
      }
      
      console.log('Generation result:', result)
      
      // Move previous current images to history
      if (currentGenerationImages.length > 0) {
        setHistoryImages(prev => [...currentGenerationImages, ...prev])
      }
      
      setGeneratedImages(prev => [...result, ...prev])
      setCurrentGenerationImages(result)
      setStatus('success')
      setGenerationProgress(`Successfully generated ${result.length} image(s)`)
      toast.success(`Generated ${result.length} image(s)`)
    } catch (error: any) {
      if (error.name === 'AbortError') {
        setStatus('cancelled')
        setGenerationProgress('Generation cancelled')
        return
      }
      
      console.error('Generation failed:', error)
      setStatus('error')
      setCurrentError(error.message || 'Failed to generate image')
      setGenerationProgress('Generation failed')
      toast.error('Failed to generate image. Please try again.')
    } finally {
      abortControllerRef.current = null
    }
  }

  const handleCancelGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setStatus('cancelled')
      setGenerationProgress('Cancelling generation...')
    }
  }

  const handleRetry = () => {
    setStatus('idle')
    setCurrentError(null)
    setGenerationProgress('')
    handleGenerate()
  }

  const handleRandomSeed = () => {
    setParams(prev => ({ ...prev, seed: Math.floor(Math.random() * 1000000) }))
  }

  const downloadAllImages = async () => {
    try {
      const displayImages = [...currentGenerationImages, ...historyImages]
      const imagesToDownload = displayImages.map(img => ({ url: img.url, id: img.id }))
      await downloadAllCloudflareImages(imagesToDownload)
      toast.success(`Downloaded ${displayImages.length} images`)
    } catch (error) {
      console.error('Download failed:', error)
      toast.error('Some downloads may have failed')
    }
  }

  const presetSizes = [
    { label: 'Square\n1024×1024', width: 1024, height: 1024 },
    { label: 'Landscape\n1216×832', width: 1216, height: 832 },
    { label: 'Portrait\n832×1216', width: 832, height: 1216 },
  ]

  const getStatusIcon = () => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 animate-spin" />
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'cancelled':
        return <StopCircle className="w-4 h-4 text-gray-500" />
      default:
        return null
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'pending':
        return 'Generating...'
      case 'success':
        return 'Generation complete'
      case 'error':
        return 'Generation failed'
      case 'cancelled':
        return 'Generation cancelled'
      default:
        return 'Ready to generate'
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Panel */}
        <div className="lg:col-span-1 space-y-4">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Text to Image ({baseModel === 'realistic' ? '真人风格' : '动漫风格'})
            </h3>
            
            {/* Status Display */}
            {(status !== 'idle' || generationProgress) && (
              <div className={`mb-4 p-3 rounded-lg border ${
                status === 'success' ? 'bg-green-50 border-green-200' :
                status === 'error' ? 'bg-red-50 border-red-200' :
                status === 'cancelled' ? 'bg-gray-50 border-gray-200' :
                'bg-blue-50 border-blue-200'
              }`}>
                <div className="flex items-center space-x-2">
                  {getStatusIcon()}
                  <span className="text-sm font-medium">{getStatusText()}</span>
                </div>
                {generationProgress && (
                  <p className="text-xs text-gray-600 mt-1">{generationProgress}</p>
                )}
                {currentError && (
                  <p className="text-xs text-red-600 mt-1">{currentError}</p>
                )}
              </div>
            )}
            
            {/* Prompt */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Prompt *
              </label>
              <textarea
                value={params.prompt}
                onChange={(e) => setParams(prev => ({ ...prev, prompt: e.target.value }))}
                placeholder="A beautiful landscape with mountains and lake at sunset..."
                className="textarea-field h-24"
                disabled={status === 'pending'}
              />
            </div>

            {/* Negative Prompt - Only for Anime Models */}
            {baseModel === 'anime' && (
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Negative Prompt
                </label>
                <textarea
                  value={params.negativePrompt}
                  onChange={(e) => setParams(prev => ({ ...prev, negativePrompt: e.target.value }))}
                  placeholder="low quality, blurry, bad anatomy, distorted..."
                  className="textarea-field h-20"
                  disabled={status === 'pending'}
                />
              </div>
            )}

            {/* Size Presets */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Aspect Ratio
              </label>
              <div className="grid grid-cols-3 gap-2">
                {presetSizes.map((size) => (
                  <button
                    key={size.label}
                    onClick={() => setParams(prev => ({ 
                      ...prev, 
                      width: size.width, 
                      height: size.height 
                    }))}
                    className={`p-3 text-xs text-center rounded-lg border transition-colors whitespace-pre-line ${
                      params.width === size.width && params.height === size.height
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                    disabled={status === 'pending'}
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
                    onClick={() => {
                      console.log('Setting numImages to:', num)
                      setParams(prev => ({ ...prev, numImages: num }))
                    }}
                    className={`p-2 text-sm rounded-lg border transition-colors ${
                      params.numImages === num
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                    disabled={status === 'pending'}
                  >
                    {num}
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-500">Current: {params.numImages} image(s)</p>
            </div>

            {/* LoRA Model Selector */}
            <LoRASelector
              value={loraConfig}
              onChange={(newConfig) => {
                console.log('LoRA config changing to:', newConfig)
                setLoraConfig(newConfig)
              }}
              baseModel={baseModel}
              disabled={status === 'pending'}
            />

            {/* Advanced Settings Toggle */}
            <button
              onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
              className="flex items-center justify-between w-full p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              disabled={status === 'pending'}
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
                    Steps: {params.steps} {baseModel === 'realistic' ? '(FLUX推荐12)' : '(动漫推荐20)'}
                  </label>
                  <input
                    type="range"
                    min={baseModel === 'realistic' ? "8" : "10"}
                    max={baseModel === 'realistic' ? "20" : "50"}
                    value={params.steps}
                    onChange={(e) => setParams(prev => ({ ...prev, steps: Number(e.target.value) }))}
                    className="slider"
                    disabled={status === 'pending'}
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{baseModel === 'realistic' ? '8' : '10'}</span>
                    <span>{baseModel === 'realistic' ? '20' : '50'}</span>
                  </div>
                </div>

                {/* CFG Scale */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">
                    CFG Scale: {params.cfgScale} {baseModel === 'realistic' ? '(FLUX推荐1)' : '(动漫推荐7)'}
                  </label>
                  <input
                    type="range"
                    min={baseModel === 'realistic' ? "0.5" : "1"}
                    max={baseModel === 'realistic' ? "3" : "20"}
                    step="0.5"
                    value={params.cfgScale}
                    onChange={(e) => setParams(prev => ({ ...prev, cfgScale: Number(e.target.value) }))}
                    className="slider"
                    disabled={status === 'pending'}
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{baseModel === 'realistic' ? '0.5' : '1'}</span>
                    <span>{baseModel === 'realistic' ? '3' : '20'}</span>
                  </div>
                </div>

                {/* Seed */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Seed (-1 for random)
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      value={params.seed}
                      onChange={(e) => setParams(prev => ({ ...prev, seed: Number(e.target.value) }))}
                      className="input-field flex-1"
                      disabled={status === 'pending'}
                    />
                    <button
                      onClick={handleRandomSeed}
                      className="btn-secondary px-3"
                      disabled={status === 'pending'}
                    >
                      <Shuffle className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Generate Button */}
            <div className="flex space-x-2">
              {status === 'pending' ? (
                <button
                  onClick={handleCancelGeneration}
                  className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                >
                  <StopCircle className="w-4 h-4" />
                  <span>Cancel</span>
                </button>
              ) : (
                <button
                  onClick={handleGenerate}
                  disabled={!params.prompt.trim()}
                  className="btn-primary flex-1 flex items-center justify-center space-x-2"
                >
                  <Play className="w-4 h-4" />
                  <span>Generate</span>
                </button>
              )}
              
              {status === 'error' && (
                <button
                  onClick={handleRetry}
                  className="btn-secondary flex items-center justify-center space-x-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Retry</span>
                </button>
              )}
            </div>

            {/* Download All Button */}
            {(currentGenerationImages.length > 0 || historyImages.length > 0) && (
              <button
                onClick={downloadAllImages}
                className="btn-secondary w-full flex items-center justify-center space-x-2"
                disabled={status === 'pending'}
              >
                <Download className="w-4 h-4" />
                <span>Download All ({currentGenerationImages.length + historyImages.length})</span>
              </button>
            )}
          </div>
        </div>

        {/* Image Gallery */}
        <div className="lg:col-span-2">
          <ImageGallery
            currentImages={currentGenerationImages}
            historyImages={historyImages}
            isLoading={status === 'pending'}
            title="Generated Images"
            onDownloadAll={downloadAllImages}
          />
        </div>
      </div>
    </div>
  )
} 