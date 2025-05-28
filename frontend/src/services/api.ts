import axios from 'axios'
import type { 
  TextToImageParams, 
  ImageToImageParams, 
  GeneratedImage, 
  ApiResponse,
  LoRAResponse
} from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api'

// For static deployment (Cloudflare Pages), use RunPod API directly
// In production mode, if we have RunPod credentials, use direct calls
const USE_RUNPOD_DIRECT = Boolean(
  (process.env.NODE_ENV === 'production' && 
   process.env.NEXT_PUBLIC_RUNPOD_API_KEY && 
   process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID) ||
  (typeof window !== 'undefined' && 
   !API_BASE_URL.includes('/api'))
)

const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY
const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID

console.log('API Configuration:', {
  USE_RUNPOD_DIRECT,
  hasRunPodKey: !!RUNPOD_API_KEY,
  hasEndpointId: !!RUNPOD_ENDPOINT_ID,
  API_BASE_URL,
  NODE_ENV: process.env.NODE_ENV,
  isProduction: process.env.NODE_ENV === 'production'
})

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

// Direct RunPod API call
async function callRunPodAPI(taskType: string, params: any, signal?: AbortSignal): Promise<any> {
  console.log('Calling RunPod API directly:', { taskType, hasKey: !!RUNPOD_API_KEY, hasEndpoint: !!RUNPOD_ENDPOINT_ID })
  
  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    throw new Error('RunPod configuration not available. Please check environment variables.')
  }

  const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`

  const runpodRequest = {
    input: {
      task_type: taskType,
      params: params,
      ...(taskType === 'switch-lora' && { lora_id: params.lora_id })
    }
  }

  console.log('RunPod request:', { url: RUNPOD_API_URL, request: runpodRequest })

  const response = await axios.post(RUNPOD_API_URL, runpodRequest, {
    headers: {
      'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      'Content-Type': 'application/json',
    },
    timeout: 300000,
    signal: signal,
  })

  console.log('RunPod response:', response.data)

  if (response.data.status === 'COMPLETED') {
    const output = response.data.output
    if (output.success) {
      return output.data
    } else {
      throw new Error(output.error || 'Generation failed')
    }
  } else {
    throw new Error(`RunPod job failed with status: ${response.data.status}`)
  }
}

// Generate text-to-image
export async function generateTextToImage(params: TextToImageParams, signal?: AbortSignal): Promise<GeneratedImage[]> {
  try {
    console.log('generateTextToImage called with USE_RUNPOD_DIRECT:', USE_RUNPOD_DIRECT)
    console.log('Requested LoRA model:', params.lora_model)
    
    // 优化：不在前端进行LoRA切换，让后端自动处理
    // 后端会检查并只在需要时进行切换
    
    if (USE_RUNPOD_DIRECT) {
      return await callRunPodAPI('text-to-image', params, signal)
    }

    const response = await api.post<ApiResponse<GeneratedImage[]>>('/generate/text-to-image', params, {
      signal: signal,
    })
    
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
export async function generateImageToImage(params: ImageToImageParams, signal?: AbortSignal): Promise<GeneratedImage[]> {
  try {
    console.log('generateImageToImage called with USE_RUNPOD_DIRECT:', USE_RUNPOD_DIRECT)
    console.log('Requested LoRA model:', params.lora_model)
    
    // 优化：不在前端进行LoRA切换，让后端自动处理
    // 后端会检查并只在需要时进行切换
    
    if (USE_RUNPOD_DIRECT) {
      // Convert image to base64 for RunPod API
      let base64Image = ''
      if (params.image instanceof File) {
        const reader = new FileReader()
        base64Image = await new Promise((resolve, reject) => {
          reader.onload = () => {
            const result = reader.result as string
            resolve(result.split(',')[1]) // Remove data URL prefix
          }
          reader.onerror = reject
          reader.readAsDataURL(params.image as File)
        })
      }

      const runpodParams = {
        prompt: params.prompt,
        negativePrompt: params.negativePrompt,
        image: base64Image,
        width: params.width,
        height: params.height,
        steps: params.steps,
        cfgScale: params.cfgScale,
        seed: params.seed,
        numImages: params.numImages,
        denoisingStrength: params.denoisingStrength,
        lora_model: params.lora_model, // 传递LoRA模型参数给后端
      }

      return await callRunPodAPI('image-to-image', runpodParams, signal)
    }

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
        signal: signal,
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

// Get available LoRA models
export async function getAvailableLoras(signal?: AbortSignal): Promise<LoRAResponse> {
  try {
    console.log('getAvailableLoras called with USE_RUNPOD_DIRECT:', USE_RUNPOD_DIRECT)
    
    if (USE_RUNPOD_DIRECT) {
      return await callRunPodAPI('get-loras', {}, signal)
    }

    const response = await api.get<ApiResponse<LoRAResponse>>('/loras', {
      signal: signal,
    })
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get LoRA models')
    }
    
    return response.data.data
  } catch (error) {
    console.error('Failed to get LoRA models:', error)
    throw error
  }
}

// Switch LoRA model
export async function switchLoraModel(loraId: string, signal?: AbortSignal): Promise<string> {
  try {
    console.log('switchLoraModel called with USE_RUNPOD_DIRECT:', USE_RUNPOD_DIRECT)
    
    if (USE_RUNPOD_DIRECT) {
      const result = await callRunPodAPI('switch-lora', { lora_id: loraId }, signal)
      return result.message || 'LoRA switched successfully'
    }

    const response = await api.post<ApiResponse<{ message: string }>>('/loras/switch', 
      { lora_id: loraId },
      {
        signal: signal,
      }
    )
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to switch LoRA model')
    }
    
    return response.data.data.message
  } catch (error) {
    console.error('Failed to switch LoRA model:', error)
    throw error
  }
} 