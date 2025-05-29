// LoRA配置类型
export interface LoRAConfig {
  [loraId: string]: number // LoRA ID -> 权重 (0-1)
}

export interface TextToImageParams {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  steps: number
  cfgScale: number
  seed: number
  numImages: number
  lora_config?: LoRAConfig // 支持多LoRA配置
  lora_model?: string // 保留兼容性
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
  lora_config?: LoRAConfig // 支持多LoRA配置
  lora_model?: string // 保留兼容性
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
  default_weight: number
  current_weight: number
}

export interface LoRAResponse {
  loras: Record<string, LoRAModel>
  current_config: LoRAConfig
} 