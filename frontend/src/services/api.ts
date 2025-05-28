import axios from 'axios'
import type { 
  TextToImageParams, 
  ImageToImageParams, 
  GeneratedImage, 
  ApiResponse 
} from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes
  headers: {
    'Content-Type': 'application/json',
  }
})

// Request interceptor
api.interceptors.request.use((config) => {
  // Add auth token if available
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// Generate text-to-image
export async function generateTextToImage(params: TextToImageParams): Promise<GeneratedImage[]> {
  try {
    const response = await api.post<ApiResponse<GeneratedImage[]>>('/generate/text-to-image', params)
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Generation failed')
    }
    
    return response.data.data
  } catch (error) {
    console.error('Text-to-image generation failed:', error)
    throw error
  }
}

// Generate image-to-image
export async function generateImageToImage(params: ImageToImageParams): Promise<GeneratedImage[]> {
  try {
    const formData = new FormData()
    
    // Add image file
    if (params.image instanceof File) {
      formData.append('image', params.image)
    }
    
    // Add other parameters
    formData.append('prompt', params.prompt)
    formData.append('negativePrompt', params.negativePrompt)
    formData.append('width', params.width.toString())
    formData.append('height', params.height.toString())
    formData.append('steps', params.steps.toString())
    formData.append('cfgScale', params.cfgScale.toString())
    formData.append('seed', params.seed.toString())
    formData.append('numImages', params.numImages.toString())
    formData.append('denoisingStrength', params.denoisingStrength.toString())
    
    const response = await api.post<ApiResponse<GeneratedImage[]>>(
      '/generate/image-to-image', 
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Generation failed')
    }
    
    return response.data.data
  } catch (error) {
    console.error('Image-to-image generation failed:', error)
    throw error
  }
}

// Upload image to storage
export async function uploadImage(file: File): Promise<string> {
  try {
    const formData = new FormData()
    formData.append('image', file)
    
    const response = await api.post<ApiResponse<{ url: string }>>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Upload failed')
    }
    
    return response.data.data.url
  } catch (error) {
    console.error('Image upload failed:', error)
    throw error
  }
}

// Download image
export async function downloadImage(url: string, filename: string): Promise<void> {
  try {
    const response = await fetch(url)
    const blob = await response.blob()
    
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  } catch (error) {
    console.error('Image download failed:', error)
    throw error
  }
}

// Get generation status
export async function getGenerationStatus(jobId: string) {
  try {
    const response = await api.get(`/generate/status/${jobId}`)
    return response.data
  } catch (error) {
    console.error('Failed to get generation status:', error)
    throw error
  }
} 