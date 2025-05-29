'use client'

import { useState, useEffect } from 'react'
import { Wand2, RotateCcw, Eye, EyeOff } from 'lucide-react'
import type { LoRAConfig } from '@/types'

interface LoRASelectorProps {
  value: LoRAConfig
  onChange: (config: LoRAConfig) => void
  disabled?: boolean
}

// Static LoRA models based on user's available models
const AVAILABLE_LORAS = {
  flux_nsfw: {
    name: 'FLUX NSFW',
    description: 'NSFW content generation model',
    defaultWeight: 1.0
  },
  UltraRealPhoto: {
    name: 'Ultra Real Photo',
    description: 'Ultra realistic photo generation',
    defaultWeight: 1.0
  },
  Chastity_Cage: {
    name: 'Chastity Cage',
    description: 'Chastity device focused generation',
    defaultWeight: 0.5
  },
  DynamicPenis: {
    name: 'Dynamic Penis',
    description: 'Dynamic male anatomy generation',
    defaultWeight: 0.5
  },
  OnOff: {
    name: 'On Off',
    description: 'Clothing on/off variations',
    defaultWeight: 0.5
  },
  Puppy_mask: {
    name: 'Puppy Mask',
    description: 'Puppy mask and pet play content',
    defaultWeight: 0.5
  },
  asianman: {
    name: 'Asian Man',
    description: 'Asian male character generation',
    defaultWeight: 0.5
  },
  'butt-and-feet': {
    name: 'Butt and Feet',
    description: 'Focus on lower body parts',
    defaultWeight: 0.5
  },
  cumshots: {
    name: 'Cumshots',
    description: 'Adult climax content generation',
    defaultWeight: 0.5
  }
}

export default function LoRASelector({ value, onChange, disabled }: LoRASelectorProps) {
  const [localConfig, setLocalConfig] = useState<LoRAConfig>(value)
  const [showInactive, setShowInactive] = useState(false)

  // 同步外部值变化
  useEffect(() => {
    setLocalConfig(value)
  }, [value])

  const handleWeightChange = (loraId: string, weight: number) => {
    const newConfig = { ...localConfig, [loraId]: weight }
    setLocalConfig(newConfig)
    onChange(newConfig)
  }

  const resetToDefaults = () => {
    const defaultConfig: LoRAConfig = {}
    Object.entries(AVAILABLE_LORAS).forEach(([id, lora]) => {
      defaultConfig[id] = lora.defaultWeight
    })
    setLocalConfig(defaultConfig)
    onChange(defaultConfig)
  }

  const getActiveLoRAs = () => {
    return Object.entries(AVAILABLE_LORAS).filter(([id]) => 
      (localConfig[id] || 0) > 0
    )
  }

  const getInactiveLoRAs = () => {
    return Object.entries(AVAILABLE_LORAS).filter(([id]) => 
      (localConfig[id] || 0) === 0
    )
  }

  const activeLoRAs = getActiveLoRAs()
  const inactiveLoRAs = getInactiveLoRAs()

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Models
        </label>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowInactive(!showInactive)}
            className="p-1 text-gray-500 hover:text-gray-700 transition-colors"
            title={showInactive ? "Hide inactive models" : "Show all models"}
          >
            {showInactive ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
          <button
            onClick={resetToDefaults}
            disabled={disabled}
            className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-50 transition-colors"
            title="Reset to defaults"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Active LoRAs */}
      {activeLoRAs.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-600 uppercase tracking-wider">
            Active Models ({activeLoRAs.length})
          </h4>
          {activeLoRAs.map(([loraId, lora]) => (
            <div key={loraId} className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <Wand2 className="w-4 h-4 text-green-600" />
                    <span className="font-medium text-gray-900">{lora.name}</span>
                    <span className="text-sm font-medium text-green-700">
                      {((localConfig[loraId] || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 mt-1">{lora.description}</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localConfig[loraId] || 0}
                  onChange={(e) => handleWeightChange(loraId, Number(e.target.value))}
                  disabled={disabled}
                  className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer slider-green"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Inactive LoRAs */}
      {(showInactive || inactiveLoRAs.length === Object.keys(AVAILABLE_LORAS).length) && (
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-600 uppercase tracking-wider">
            {showInactive ? `Inactive Models (${inactiveLoRAs.length})` : 'Available Models'}
          </h4>
          {inactiveLoRAs.map(([loraId, lora]) => (
            <div key={loraId} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <Wand2 className="w-4 h-4 text-gray-400" />
                    <span className="font-medium text-gray-700">{lora.name}</span>
                    <span className="text-sm font-medium text-gray-500">
                      {((localConfig[loraId] || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{lora.description}</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localConfig[loraId] || 0}
                  onChange={(e) => handleWeightChange(loraId, Number(e.target.value))}
                  disabled={disabled}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Info */}
      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <Wand2 className="w-4 h-4 text-blue-600 mt-0.5" />
          <div className="text-xs text-blue-700">
            <p className="font-medium mb-1">💡 LoRA Usage Tips:</p>
            <ul className="space-y-1 text-blue-600">
              <li>• Higher weights = stronger influence on generation</li>
              <li>• Multiple LoRAs will be combined together</li>
              <li>• Set to 0% to disable a LoRA completely</li>
              <li>• Default weights are optimized for best results</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
} 