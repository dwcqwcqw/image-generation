'use client'

import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { toast } from 'react-hot-toast'
import Image from 'next/image'
import { 
  Play, 
  Upload, 
  RefreshCw, 
  Settings, 
  ChevronDown,
  ChevronUp,
  Shuffle,
  X,
  Download,
  StopCircle,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react'
import ImageGallery from './ImageGallery'
import LoRASelector from './LoRASelector'
import BaseModelSelector from './BaseModelSelector'
import { generateImageToImage } from '@/services/api'
import { downloadAllImages as downloadAllImagesUtil } from '@/utils/imageProxy'
import type { ImageToImageParams, GeneratedImage, LoRAConfig, BaseModelType } from '@/types'

type GenerationStatus = 'idle' | 'pending' | 'success' | 'error' | 'cancelled'

export default function ImageToImagePanel() {
  const [status, setStatus] = useState<GenerationStatus>('idle')
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(true)
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  const [currentGenerationImages, setCurrentGenerationImages] = useState<GeneratedImage[]>([])
  const [historyImages, setHistoryImages] = useState<GeneratedImage[]>([])
  const [sourceImage, setSourceImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [currentError, setCurrentError] = useState<string | null>(null)
  const [generationProgress, setGenerationProgress] = useState<string>('')
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // 默认LoRA配置 - 简化为只使用FLUX NSFW
  const defaultLoRAConfig: LoRAConfig = {
    flux_nsfw: 1.0
  }
  
  const [params, setParams] = useState<ImageToImageParams>({
    prompt: '',
    negativePrompt: 'low quality, blurry, bad anatomy, deformed hands, extra fingers, missing fingers, deformed limbs, extra limbs, bad proportions, malformed genitals, watermark',
    image: '',
    width: 512,
    height: 512,
    steps: 20,
    cfgScale: 7.0,
    seed: -1,
    numImages: 1,
    denoisingStrength: 0.7,
    baseModel: 'realistic',
    lora_config: defaultLoRAConfig,
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSourceImage(file)
      setParams(prev => ({ ...prev, image: file }))
      
      // Create preview
      const reader = new FileReader()
      reader.onload = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  })

  const handleGenerate = async () => {
    setStatus('pending')
    setCurrentError(null)
    setGenerationProgress('Preparing generation...')
    setCurrentGenerationImages([]) // 清空当前生成的图片
    
    const abortController = new AbortController()
    abortControllerRef.current = abortController
    
    try {
      // Convert file to base64
      if (!sourceImage) return
      
      const reader = new FileReader()
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(sourceImage)
      })
      
      const base64Image = await base64Promise
      
      const requestParams = {
        ...params,
        image: base64Image
      }
      
      setGenerationProgress('Generating images...')
      const result = await generateImageToImage(requestParams, abortController.signal)
      
      if (abortController.signal.aborted) {
        setStatus('cancelled')
        setGenerationProgress('Generation cancelled')
        return
      }
      
      // 将之前的current images移到history
      if (currentGenerationImages.length > 0) {
        setHistoryImages(prev => [...currentGenerationImages, ...prev])
      }
      
      setGeneratedImages(prev => [...result, ...prev])
      setCurrentGenerationImages(result) // 设置新生成的图片为current
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

  const removeImage = () => {
    setSourceImage(null)
    setImagePreview(null)
    setParams(prev => ({ ...prev, image: '' }))
  }

  const downloadAllImages = async () => {
    try {
      const displayImages = [...currentGenerationImages, ...historyImages]
      const imagesToDownload = displayImages.map(img => ({ url: img.url, id: img.id }))
      await downloadAllImagesUtil(imagesToDownload)
      toast.success(`Downloaded ${displayImages.length} images`)
    } catch (error) {
      console.error('Download failed:', error)
      toast.error('Some downloads may have failed')
    }
  }

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
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Image to Image</h3>
            
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
            
            {/* Image Upload */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Source Image *
              </label>
              
              {imagePreview ? (
                <div className="relative">
                  <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                    <Image
                      src={imagePreview}
                      alt="Source image"
                      fill
                      className="object-cover"
                      sizes="300px"
                    />
                  </div>
                  {status !== 'pending' && (
                    <button
                      onClick={removeImage}
                      className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ) : (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-300 hover:border-gray-400'
                  } ${status === 'pending' ? 'opacity-50 pointer-events-none' : ''}`}
                >
                  <input {...getInputProps()} disabled={status === 'pending'} />
                  <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">
                    {isDragActive
                      ? 'Drop the image here...'
                      : 'Drag & drop an image, or click to select'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports: JPEG, PNG, WebP (max 10MB)
                  </p>
                </div>
              )}
            </div>

            {/* Prompt */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Positive Prompt *
              </label>
              <textarea
                value={params.prompt}
                onChange={(e) => setParams(prev => ({ ...prev, prompt: e.target.value }))}
                placeholder="Transform this image into..."
                className="textarea-field h-24"
                disabled={status === 'pending'}
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
                disabled={status === 'pending'}
              />
            </div>

            {/* Size Presets */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Output Size
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
                    onClick={() => setParams(prev => ({ ...prev, numImages: num }))}
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
            </div>

            {/* Denoising Strength */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Denoising Strength: {params.denoisingStrength}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={params.denoisingStrength}
                onChange={(e) => setParams(prev => ({ ...prev, denoisingStrength: Number(e.target.value) }))}
                className="slider"
                disabled={status === 'pending'}
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>0.1 (Subtle)</span>
                <span>1.0 (Strong)</span>
              </div>
              <p className="text-xs text-gray-500">
                Lower values preserve more of the original image
              </p>
            </div>

            {/* LoRA Model Selector */}
            <LoRASelector
              value={params.lora_config || defaultLoRAConfig}
              onChange={(loraConfig) => setParams(prev => ({ ...prev, lora_config: loraConfig }))}
              baseModel={params.baseModel}
              disabled={status === 'pending'}
            />

            {/* Base Model Selector */}
            <BaseModelSelector
              value={params.baseModel}
              onChange={(baseModel) => {
                // 根据基础模型自动配置对应的LoRA
                const newLoRAConfig: LoRAConfig = baseModel === 'realistic' 
                  ? { flux_nsfw: 1.0 } 
                  : { gayporn: 1.0 }
                
                setParams(prev => ({ 
                  ...prev, 
                  baseModel,
                  lora_config: newLoRAConfig
                }))
              }}
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
                    Steps: {params.steps}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={params.steps}
                    onChange={(e) => setParams(prev => ({ ...prev, steps: Number(e.target.value) }))}
                    className="slider"
                    disabled={status === 'pending'}
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
                    disabled={status === 'pending'}
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>1</span>
                    <span>20</span>
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

            {/* Action Buttons */}
            <div className="space-y-3 pt-4">
              {status === 'pending' ? (
                <button
                  onClick={handleCancelGeneration}
                  className="btn-secondary w-full flex items-center justify-center space-x-2"
                >
                  <StopCircle className="w-4 h-4" />
                  <span>Cancel Generation</span>
                </button>
              ) : (
                <button
                  onClick={handleGenerate}
                  disabled={!params.prompt.trim() || !sourceImage}
                  className="btn-primary w-full flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play className="w-4 h-4" />
                  <span>Transform Image</span>
                </button>
              )}

              {status === 'error' && (
                <button
                  onClick={handleRetry}
                  className="btn-secondary w-full flex items-center justify-center space-x-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Try Again</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          {(currentGenerationImages.length > 0 || historyImages.length > 0) ? (
            <ImageGallery 
              currentImages={currentGenerationImages}
              historyImages={historyImages}
              isLoading={status === 'pending'}
              onDownloadAll={downloadAllImages}
            />
          ) : (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Generated Images</h3>
              <div className="text-center py-12 text-gray-500">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-gray-400" />
                </div>
                <p>No images generated yet</p>
                <p className="text-sm">Upload an image and enter a prompt to start</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 