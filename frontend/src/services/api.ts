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

// Direct RunPod API call with queue handling
async function callRunPodAPI(taskType: string, params: any, signal?: AbortSignal): Promise<any> {
  console.log('Calling RunPod API directly:', { taskType, hasKey: !!RUNPOD_API_KEY, hasEndpoint: !!RUNPOD_ENDPOINT_ID })
  
  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    throw new Error('RunPod configuration not available. Please check environment variables.')
  }

  const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`

  // 🚀 Fix: Flatten parameters structure for direct backend access
  const runpodRequest = {
    input: {
      task_type: taskType,
      // Flatten all parameters directly to input level
      ...params,
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

  // 🚀 优化：处理队列状态，避免抛出错误
  if (response.data.status === 'COMPLETED') {
    const output = response.data.output
    if (output.success) {
      return output.data
    } else {
      throw new Error(output.error || 'Generation failed')
    }
  } else if (response.data.status === 'IN_QUEUE') {
    // 🔄 处理队列状态 - 轮询等待完成
    console.log('Job is in queue, polling for completion...')
    const jobId = response.data.id
    
    if (!jobId) {
      throw new Error('Job queued but no job ID received')
    }
    
    return await pollRunPodJob(jobId, signal)
  } else if (response.data.status === 'IN_PROGRESS') {
    // 🔄 处理进行中状态 - 轮询等待完成
    console.log('Job is in progress, polling for completion...')
    const jobId = response.data.id
    
    if (!jobId) {
      throw new Error('Job in progress but no job ID received')
    }
    
    return await pollRunPodJob(jobId, signal)
  } else {
    throw new Error(`RunPod job failed with status: ${response.data.status}`)
  }
}

// 轮询RunPod作业状态直到完成
async function pollRunPodJob(jobId: string, signal?: AbortSignal): Promise<any> {
  const POLL_INTERVAL = 2000 // 2秒轮询间隔
  const MAX_POLL_TIME = 300000 // 5分钟最大等待时间
  const startTime = Date.now()
  
  const RUNPOD_STATUS_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`
  
  while (Date.now() - startTime < MAX_POLL_TIME) {
    if (signal?.aborted) {
      throw new Error('Request was aborted')
    }
    
    try {
      console.log(`Polling job ${jobId} status...`)
      
      const statusResponse = await axios.get(RUNPOD_STATUS_URL, {
        headers: {
          'Authorization': `Bearer ${RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        timeout: 10000, // 10秒超时
        signal: signal,
      })
      
      const status = statusResponse.data.status
      console.log(`Job ${jobId} status: ${status}`)
      
      if (status === 'COMPLETED') {
        const output = statusResponse.data.output
        if (output.success) {
          return output.data
        } else {
          throw new Error(output.error || 'Generation failed')
        }
      } else if (status === 'FAILED') {
        const errorMsg = statusResponse.data.error || 'Job failed'
        throw new Error(`RunPod job failed: ${errorMsg}`)
      } else if (status === 'CANCELLED') {
        throw new Error('RunPod job was cancelled')
      }
      // 继续轮询 IN_QUEUE 和 IN_PROGRESS 状态
      
    } catch (error) {
      if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
        console.log('Status poll timeout, retrying...')
      } else {
        console.error('Error polling job status:', error)
        throw error
      }
    }
    
    // 等待轮询间隔
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL))
  }
  
  throw new Error('Job polling timeout - please try again or check RunPod status')
}

// Generate text-to-image
export async function generateTextToImage(params: TextToImageParams, signal?: AbortSignal): Promise<GeneratedImage[]> {
  try {
    console.log('generateTextToImage called with USE_RUNPOD_DIRECT:', USE_RUNPOD_DIRECT)
    console.log('Full parameters being sent:', params)
    console.log('Requested numImages:', params.numImages)
    console.log('Requested baseModel:', params.baseModel)
    console.log('Requested LoRA config:', params.lora_config)
    
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
    console.log('[Download] Starting download:', url)
    
    // 方法1: 尝试创建下载链接（最可靠）
    try {
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
      
      // 添加到DOM，点击，然后移除
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      console.log('[Download] Direct link download initiated')
      return
    } catch (linkError) {
      console.log('[Download] Direct link failed, trying fetch...', linkError)
    }
    
    // 方法2: 使用fetch下载
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'image/*',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const blob = await response.blob()
      console.log('[Download] Fetch successful, blob size:', blob.size, 'bytes')
      
      // 创建blob URL并下载
      const blobUrl = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = blobUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(blobUrl)
      
      console.log('[Download] Fetch download successful')
      return
    } catch (fetchError) {
      console.log('[Download] Fetch failed, trying canvas method...', fetchError)
    }
    
    // 方法3: 使用canvas转换（适用于CORS限制的图片）
    try {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = url
      })
      
      // 创建canvas并绘制图片
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      canvas.width = img.width
      canvas.height = img.height
      
      if (ctx) {
        ctx.drawImage(img, 0, 0)
        
        // 转换为blob并下载
        canvas.toBlob((blob) => {
          if (blob) {
            const blobUrl = window.URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = blobUrl
            link.download = filename
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            window.URL.revokeObjectURL(blobUrl)
            console.log('[Download] Canvas download successful')
          } else {
            throw new Error('Failed to create blob from canvas')
          }
        }, 'image/png')
        return
      }
    } catch (canvasError) {
      console.log('[Download] Canvas method failed:', canvasError)
    }
    
    // 方法4: 最后回退 - 在新窗口打开
    console.log('[Download] All methods failed, opening in new window')
    const newWindow = window.open(url, '_blank', 'noopener,noreferrer')
    if (newWindow) {
      // 给用户一些提示
      setTimeout(() => {
        alert('图片已在新窗口打开，请右键点击图片选择"图片另存为"来下载')
      }, 1000)
    } else {
      throw new Error('无法打开新窗口，请检查浏览器弹窗设置')
    }
    
  } catch (error) {
    console.error('[Download] All download methods failed:', error)
    throw new Error(`下载失败: ${error instanceof Error ? error.message : '未知错误'}。请尝试右键点击图片选择"图片另存为"`)
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