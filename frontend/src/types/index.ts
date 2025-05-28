export interface TextToImageParams {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  steps: number
  cfgScale: number
  seed: number
  numImages: number
  lora_model?: string
}

export interface ImageToImageParams {
  prompt: string
  negativePrompt: string
  image: string | File
  width: number
  height: number
  steps: number
  cfgScale: number
  seed: number
  numImages: number
  denoisingStrength: number
  lora_model?: string
}

export interface GeneratedImage {
  id: string
  url: string
  prompt: string
  negativePrompt?: string
  seed: number
  width: number
  height: number
  steps: number
  cfgScale: number
  createdAt: string
  type: 'text-to-image' | 'image-to-image'
}

export interface ApiResponse<T> {
  success: boolean
  data: T
  error?: string
}

export interface GenerationJob {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress?: number
  result?: GeneratedImage[]
  error?: string
}

// LoRA模型接口
export interface LoRAModel {
  name: string
  description: string
  is_current: boolean
}

export interface LoRAResponse {
  loras: Record<string, LoRAModel>
  current: string
} 