'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Download, Eye, Copy, Trash2, RefreshCw, Archive, Clock } from 'lucide-react'
import { toast } from 'react-hot-toast'
import type { GeneratedImage } from '@/types'
import { 
  getCloudflareImageUrl, 
  downloadCloudflareImage, 
  downloadAllCloudflareImages,
  debugImageUrl,
  getProxyImageUrl
} from '@/utils/cloudflareImageProxy'

interface ImageGalleryProps {
  currentImages?: GeneratedImage[]    // 当前任务生成的图片
  historyImages?: GeneratedImage[]    // 历史生成的图片
  isLoading?: boolean
  title?: string
  onDownloadAll?: () => void         // download all 回调函数
}

export default function ImageGallery({ 
  currentImages = [], 
  historyImages = [], 
  isLoading, 
  title = "Generated Images",
  onDownloadAll 
}: ImageGalleryProps) {
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null)
  const [showTab, setShowTab] = useState<'current' | 'all'>('current')

  const allImages = [...currentImages, ...historyImages]
  const displayImages = showTab === 'current' ? currentImages : allImages

  const handleDownload = async (image: GeneratedImage) => {
    try {
      const filename = `ai-generated-${image.id}.png`
      await downloadCloudflareImage(image.url, filename)
      toast.success('Image downloaded successfully')
    } catch (error) {
      console.error('Download failed:', error)
      toast.error('Download failed, but opened in new window')
    }
  }

  const handleDownloadAll = async () => {
    if (onDownloadAll) {
      onDownloadAll()
    } else {
      try {
        const imagesToDownload = displayImages.map(img => ({ url: img.url, id: img.id }))
        await downloadAllCloudflareImages(imagesToDownload)
        toast.success(`Downloaded ${displayImages.length} images`)
      } catch (error) {
        console.error('Batch download failed:', error)
        toast.error('Some downloads may have failed')
      }
    }
  }

  const copyPrompt = (prompt: string) => {
    navigator.clipboard.writeText(prompt)
    toast.success('Prompt copied to clipboard')
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const isCurrentImage = (imageId: string) => {
    return currentImages.some(img => img.id === imageId)
  }

  if (isLoading && allImages.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="flex flex-col items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 text-primary-600 animate-spin mb-4" />
          <p className="text-gray-600">Generating your images...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
      </div>
    )
  }

  if (allImages.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mb-4">
            <Eye className="w-8 h-8 text-gray-400" />
          </div>
          <p className="text-gray-600">No images generated yet</p>
          <p className="text-sm text-gray-500 mt-2">Your generated images will appear here</p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      {/* Header with title, tabs, and download all button */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          
          {/* Tab switcher */}
          {historyImages.length > 0 && (
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setShowTab('current')}
                className={`px-3 py-1 text-sm rounded-md transition-colors flex items-center space-x-1 ${
                  showTab === 'current'
                    ? 'bg-white text-primary-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Clock className="w-3 h-3" />
                <span>Current ({currentImages.length})</span>
              </button>
              <button
                onClick={() => setShowTab('all')}
                className={`px-3 py-1 text-sm rounded-md transition-colors flex items-center space-x-1 ${
                  showTab === 'all'
                    ? 'bg-white text-primary-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Archive className="w-3 h-3" />
                <span>All ({allImages.length})</span>
              </button>
            </div>
          )}
        </div>

        {/* Download all button and image count */}
        <div className="flex items-center space-x-3">
          <span className="text-sm text-gray-500">{displayImages.length} image(s)</span>
          
          {displayImages.length > 0 && (
            <button
              onClick={handleDownloadAll}
              className="btn-secondary flex items-center space-x-2 px-3 py-2 text-sm"
            >
              <Download className="w-4 h-4" />
              <span>Download All</span>
            </button>
          )}
        </div>
      </div>

      {/* Images grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading && (
          <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
            <RefreshCw className="w-6 h-6 text-gray-400 animate-spin" />
          </div>
        )}
        
        {displayImages.map((image) => (
          <div key={image.id} className="group relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
            {/* Image type badge */}
            <div className="absolute top-2 left-2 z-10">
              {isCurrentImage(image.id) ? (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200">
                  <Clock className="w-3 h-3 mr-1" />
                  Current
                </span>
              ) : (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600 border border-gray-200">
                  <Archive className="w-3 h-3 mr-1" />
                  History
                </span>
              )}
            </div>

            <Image
              src={getCloudflareImageUrl(image.url)}
              alt={image.prompt}
              fill
              className="object-cover transition-transform duration-300 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
              onError={(e) => {
                console.error('Image load error for:', image.url)
                debugImageUrl(image.url)
                const target = e.target as HTMLImageElement
                if (target.src === getCloudflareImageUrl(image.url)) {
                  console.log('Trying proxy URL as fallback')
                  target.src = getProxyImageUrl(image.url)
                } else if (target.src !== image.url) {
                  console.log('Trying original URL as final fallback')
                  target.src = image.url
                }
              }}
            />
            
            {/* Overlay */}
            <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-all duration-300 flex items-end">
              <div className="p-3 w-full transform translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setSelectedImage(image)}
                    className="p-2 bg-white bg-opacity-90 hover:bg-opacity-100 rounded-lg transition-all duration-200"
                    title="View details"
                  >
                    <Eye className="w-4 h-4 text-gray-700" />
                  </button>
                  
                  <button
                    onClick={() => handleDownload(image)}
                    className="p-2 bg-white bg-opacity-90 hover:bg-opacity-100 rounded-lg transition-all duration-200"
                    title="Download"
                  >
                    <Download className="w-4 h-4 text-gray-700" />
                  </button>
                  
                  <button
                    onClick={() => copyPrompt(image.prompt)}
                    className="p-2 bg-white bg-opacity-90 hover:bg-opacity-100 rounded-lg transition-all duration-200"
                    title="Copy prompt"
                  >
                    <Copy className="w-4 h-4 text-gray-700" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Image Detail Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-4xl max-h-[90vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-3">
                  <h4 className="text-xl font-semibold text-gray-900">Image Details</h4>
                  {isCurrentImage(selectedImage.id) ? (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      <Clock className="w-3 h-3 mr-1" />
                      Current
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      <Archive className="w-3 h-3 mr-1" />
                      History
                    </span>
                  )}
                </div>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
                >
                  ×
                </button>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <Image
                    src={getCloudflareImageUrl(selectedImage.url)}
                    alt={selectedImage.prompt}
                    fill
                    className="object-cover"
                    sizes="(max-width: 1024px) 100vw, 50vw"
                  />
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Prompt
                    </label>
                    <p className="text-sm text-gray-900 bg-gray-50 p-3 rounded-lg">
                      {selectedImage.prompt}
                    </p>
                  </div>
                  
                  {selectedImage.negativePrompt && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Negative Prompt
                      </label>
                      <p className="text-sm text-gray-900 bg-gray-50 p-3 rounded-lg">
                        {selectedImage.negativePrompt}
                      </p>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">Size:</span>
                      <span className="ml-2 text-gray-900">
                        {selectedImage.width} × {selectedImage.height}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Steps:</span>
                      <span className="ml-2 text-gray-900">{selectedImage.steps}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">CFG Scale:</span>
                      <span className="ml-2 text-gray-900">{selectedImage.cfgScale}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Seed:</span>
                      <span className="ml-2 text-gray-900">{selectedImage.seed}</span>
                    </div>
                    <div className="col-span-2">
                      <span className="font-medium text-gray-700">Created:</span>
                      <span className="ml-2 text-gray-900">
                        {formatTimestamp(selectedImage.createdAt)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex space-x-3 pt-4">
                    <button
                      onClick={() => handleDownload(selectedImage)}
                      className="btn-primary flex-1"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                    <button
                      onClick={() => copyPrompt(selectedImage.prompt)}
                      className="btn-secondary flex-1"
                    >
                      <Copy className="w-4 h-4 mr-2" />
                      Copy Prompt
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 