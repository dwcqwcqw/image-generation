'use client'

import { useState, useEffect } from 'react'
import { Wand2, RotateCcw, Eye, EyeOff } from 'lucide-react'
import type { LoRAConfig } from '@/types'

interface LoRASelectorProps {
  value: LoRAConfig
  onChange: (config: LoRAConfig) => void
  disabled?: boolean
}

// Static LoRA models - 简化为只显示FLUX NSFW
const AVAILABLE_LORAS = {
  flux_nsfw: {
    name: 'FLUX NSFW',
    description: 'NSFW content generation model',
    defaultWeight: 1.0
  }
}

export default function LoRASelector({ value, onChange, disabled }: LoRASelectorProps) {
  const [localConfig, setLocalConfig] = useState<LoRAConfig>(value)

  // 同步外部值变化
  useEffect(() => {
    setLocalConfig(value)
  }, [value])

  const handleToggle = (enabled: boolean) => {
    const newConfig = { flux_nsfw: enabled ? 1.0 : 0.0 }
    setLocalConfig(newConfig)
    onChange(newConfig)
  }

  const isEnabled = (localConfig.flux_nsfw || 0) > 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Model
        </label>
      </div>

      {/* FLUX NSFW Toggle */}
      <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <Wand2 className={`w-4 h-4 ${isEnabled ? 'text-green-600' : 'text-gray-400'}`} />
              <span className="font-medium text-gray-900">FLUX NSFW</span>
              <span className={`text-sm font-medium ${isEnabled ? 'text-green-700' : 'text-gray-500'}`}>
                {isEnabled ? 'ON' : 'OFF'}
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1">NSFW content generation model</p>
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
            <p className="font-medium mb-1">💡 LoRA Info:</p>
            <p className="text-blue-600">
              Toggle to enable/disable NSFW content generation. 
              When enabled, the model will be more responsive to adult content prompts.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 