'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Download, Eye, Copy, Trash2, RefreshCw } from 'lucide-react'
import { toast } from 'react-hot-toast'
import type { GeneratedImage } from '@/types'
import { downloadImage } from '@/services/api'

interface ImageGalleryProps {
  images: GeneratedImage[]
  isLoading?: boolean
  title?: string
}

export default function ImageGallery({ images, isLoading, title = "Images" }: ImageGalleryProps) {
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null)

  const handleDownload = async (image: GeneratedImage) => {
    try {
      const filename = `ai-generated-${image.id}.png`
      await downloadImage(image.url, filename)
      toast.success('Image downloaded successfully')
    } catch (error) {
      toast.error('Failed to download image')
    }
  }

  const copyPrompt = (prompt: string) => {
    navigator.clipboard.writeText(prompt)
    toast.success('Prompt copied to clipboard')
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  if (isLoading && images.length === 0) {
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

  if (images.length === 0) {
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
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <span className="text-sm text-gray-500">{images.length} image(s)</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading && (
          <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
            <RefreshCw className="w-6 h-6 text-gray-400 animate-spin" />
          </div>
        )}
        
        {images.map((image) => (
          <div key={image.id} className="group relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <Image
              src={image.url}
              alt={image.prompt}
              fill
              className="object-cover transition-transform duration-300 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
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
                <h4 className="text-xl font-semibold text-gray-900">Image Details</h4>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <Image
                    src={selectedImage.url}
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