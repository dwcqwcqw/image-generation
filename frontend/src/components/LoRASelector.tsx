'use client'

import { useState, useEffect } from 'react'
import { Wand2 } from 'lucide-react'
import type { LoRAConfig, BaseModelType } from '@/types'
import { BASE_MODELS } from './BaseModelSelector'

interface LoRASelectorProps {
  value: LoRAConfig
  onChange: (config: LoRAConfig) => void
  baseModel: BaseModelType
  disabled?: boolean
}

export default function LoRASelector({ value, onChange, baseModel, disabled }: LoRASelectorProps) {
  const [localConfig, setLocalConfig] = useState<LoRAConfig>(value)

  // 同步外部值变化
  useEffect(() => {
    setLocalConfig(value)
  }, [value])

  // 根据基础模型获取对应的LoRA信息
  const getLoRAInfo = () => {
    const modelConfig = BASE_MODELS[baseModel]
    return {
      id: baseModel === 'realistic' ? 'flux_nsfw' : 'gayporn',
      name: modelConfig.loraName,
      description: baseModel === 'realistic' 
        ? 'NSFW真人内容生成模型' 
        : 'NSFW动漫内容生成模型'
    }
  }

  const loraInfo = getLoRAInfo()
  const loraKey = loraInfo.id

  const handleToggle = (enabled: boolean) => {
    const newConfig = { [loraKey]: enabled ? 1.0 : 0.0 }
    setLocalConfig(newConfig)
    onChange(newConfig)
  }

  const isEnabled = (localConfig[loraKey] || 0) > 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          LoRA 增强模型
        </label>
      </div>

      {/* LoRA Toggle */}
      <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <Wand2 className={`w-4 h-4 ${isEnabled ? 'text-green-600' : 'text-gray-400'}`} />
              <span className="font-medium text-gray-900">{loraInfo.name}</span>
              <span className={`text-sm font-medium ${isEnabled ? 'text-green-700' : 'text-gray-500'}`}>
                {isEnabled ? 'ON' : 'OFF'}
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1">{loraInfo.description}</p>
          </div>
          
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only"
              checked={isEnabled}
              onChange={(e) => handleToggle(e.target.checked)}
              disabled={disabled}
            />
            <div className={`w-11 h-6 rounded-full transition-colors duration-200 ease-in-out ${
              isEnabled ? 'bg-green-600' : 'bg-gray-300'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
              <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform duration-200 ease-in-out ${
                isEnabled ? 'translate-x-5' : 'translate-x-0'
              } mt-0.5 ml-0.5`}></div>
            </div>
          </label>
        </div>
      </div>

      {/* Info */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <Wand2 className="w-4 h-4 text-blue-600 mt-0.5" />
          <div className="text-xs text-blue-700">
            <p className="font-medium mb-1">💡 LoRA 说明:</p>
            <p className="text-blue-600">
              LoRA模型会根据选择的基础模型风格自动配置。
              开启后可以生成更符合预期的内容效果。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 