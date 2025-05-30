'use client'

import { User, Sparkles } from 'lucide-react'
import type { BaseModelType } from '@/types'

interface BaseModelSelectorProps {
  value: BaseModelType
  onChange: (baseModel: BaseModelType) => void
  disabled?: boolean
}

export default function BaseModelSelector({ value, onChange, disabled = false }: BaseModelSelectorProps) {
  const models = [
    {
      id: 'realistic' as BaseModelType,
      name: 'Realistic Style',
      description: 'Generate photorealistic human images with detailed anatomy',
      icon: User,
      gradient: 'from-blue-500 to-purple-600'
    },
    {
      id: 'anime' as BaseModelType,
      name: 'Anime Style',
      description: 'Create stylized anime and cartoon-style illustrations',
      icon: Sparkles,
      gradient: 'from-pink-500 to-orange-500'
    }
  ]

  const handleModelChange = (modelId: BaseModelType) => {
    console.log('BaseModelSelector: Changing from', value, 'to', modelId)
    if (!disabled && onChange) {
      onChange(modelId)
    }
  }

  console.log('BaseModelSelector: Current value is', value)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {models.map((model) => {
        const IconComponent = model.icon
        const isSelected = value === model.id
        
        return (
          <button
            key={model.id}
            onClick={() => handleModelChange(model.id)}
            disabled={disabled}
            className={`relative p-6 rounded-xl border-2 transition-all duration-200 text-left ${
              isSelected
                ? 'border-primary-500 bg-primary-50 shadow-lg transform scale-[1.02]'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            {/* Selection indicator */}
            {isSelected && (
              <div className="absolute top-3 right-3">
                <div className="w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
              </div>
            )}
            
            {/* Icon with gradient background */}
            <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${model.gradient} flex items-center justify-center mb-4`}>
              <IconComponent className="w-6 h-6 text-white" />
            </div>
            
            {/* Content */}
            <div>
              <h3 className={`text-lg font-semibold mb-2 ${
                isSelected ? 'text-primary-700' : 'text-gray-900'
              }`}>
                {model.name}
              </h3>
              <p className={`text-sm ${
                isSelected ? 'text-primary-600' : 'text-gray-600'
              }`}>
                {model.description}
              </p>
            </div>
            
            {/* Debug info */}
            <div className="absolute bottom-2 right-2 text-xs text-gray-400">
              {isSelected ? '✓' : '○'} {model.id}
            </div>
          </button>
        )
      })}
    </div>
  )
} 