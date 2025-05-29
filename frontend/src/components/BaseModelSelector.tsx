'use client'

import { useState, useEffect } from 'react'
import { User, Sparkles } from 'lucide-react'
import type { BaseModelType, BaseModelConfig } from '@/types'

interface BaseModelSelectorProps {
  value: BaseModelType
  onChange: (modelType: BaseModelType) => void
  disabled?: boolean
}

// 基础模型配置 - 用户友好的显示名称
const BASE_MODELS: Record<BaseModelType, BaseModelConfig> = {
  realistic: {
    type: 'realistic',
    name: '真人风格',
    description: '生成真实人物照片风格的图像',
    basePath: '/runpod-volume/flux_base',
    loraPath: '/runpod-volume/lora/flux_nsfw',
    loraName: 'FLUX NSFW'
  },
  anime: {
    type: 'anime',
    name: '动漫风格', 
    description: '生成日式动漫插画风格的图像',
    basePath: '/runpod-volume/cartoon/waiNSFWIllustrious_v130.safetensors',
    loraPath: '/runpod-volume/cartoon/lora/Gayporn.safetensor',
    loraName: 'Gayporn'
  }
}

export default function BaseModelSelector({ value, onChange, disabled }: BaseModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<BaseModelType>(value)

  // 同步外部值变化
  useEffect(() => {
    setSelectedModel(value)
  }, [value])

  const handleModelChange = (modelType: BaseModelType) => {
    setSelectedModel(modelType)
    onChange(modelType)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          基础模型风格
        </label>
      </div>

      {/* 模型选择卡片 */}
      <div className="grid grid-cols-2 gap-3">
        {Object.entries(BASE_MODELS).map(([modelType, config]) => {
          const isSelected = selectedModel === modelType
          const IconComponent = modelType === 'realistic' ? User : Sparkles
          
          return (
            <button
              key={modelType}
              onClick={() => handleModelChange(modelType as BaseModelType)}
              disabled={disabled}
              className={`p-4 rounded-lg border-2 transition-all duration-200 text-left ${
                isSelected
                  ? 'border-primary-500 bg-primary-50 shadow-sm'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-full ${
                  isSelected ? 'bg-primary-100' : 'bg-gray-100'
                }`}>
                  <IconComponent className={`w-5 h-5 ${
                    isSelected ? 'text-primary-600' : 'text-gray-600'
                  }`} />
                </div>
                
                <div className="flex-1">
                  <h3 className={`font-medium ${
                    isSelected ? 'text-primary-900' : 'text-gray-900'
                  }`}>
                    {config.name}
                  </h3>
                  <p className={`text-xs mt-1 ${
                    isSelected ? 'text-primary-700' : 'text-gray-600'
                  }`}>
                    {config.description}
                  </p>
                </div>
                
                {/* 选中状态指示器 */}
                <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                  isSelected 
                    ? 'border-primary-500 bg-primary-500' 
                    : 'border-gray-300'
                }`}>
                  {isSelected && (
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      {/* 当前选择的模型信息 */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          {selectedModel === 'realistic' ? (
            <User className="w-4 h-4 text-blue-600 mt-0.5" />
          ) : (
            <Sparkles className="w-4 h-4 text-blue-600 mt-0.5" />
          )}
          <div className="text-xs text-blue-700">
            <p className="font-medium mb-1">
              ✨ 当前选择: {BASE_MODELS[selectedModel].name}
            </p>
            <p className="text-blue-600">
              {BASE_MODELS[selectedModel].description}
              ，已自动配置对应的LoRA模型。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// 导出配置供其他组件使用
export { BASE_MODELS } 