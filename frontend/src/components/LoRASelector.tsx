'use client'

import { useState } from 'react'
import { Wand2, CheckCircle } from 'lucide-react'
import type { LoRAModel } from '@/types'

interface LoRASelectorProps {
  value: string
  onChange: (loraId: string) => void
  disabled?: boolean
}

// Static LoRA options based on user's available models
const STATIC_LORAS: Record<string, LoRAModel> = {
  'flux-nsfw': {
    name: 'FLUX NSFW',
    description: 'NSFW content generation model with enhanced capabilities',
    is_current: false
  }
}

export default function LoRASelector({ value, onChange, disabled }: LoRASelectorProps) {
  const [selectedLora, setSelectedLora] = useState(value || 'flux-nsfw')

  const handleLoraChange = (loraId: string) => {
    if (loraId === selectedLora || disabled) return
    
    setSelectedLora(loraId)
    onChange(loraId)
    // Note: Actual LoRA switching will happen when generating images
  }

  const loraEntries = Object.entries(STATIC_LORAS)

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        LoRA Model
      </label>

      <div className="space-y-2">
        {loraEntries.map(([loraId, lora]) => (
          <button
            key={loraId}
            onClick={() => handleLoraChange(loraId)}
            disabled={disabled}
            className={`w-full p-3 text-left rounded-lg border transition-all duration-200 ${
              selectedLora === loraId
                ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
                : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            } ${
              disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <Wand2 className="w-4 h-4 text-primary-600" />
                  <span className="font-medium text-gray-900">{lora.name}</span>
                  {selectedLora === loraId && (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  )}
                </div>
                <p className="text-xs text-gray-600 mt-1">{lora.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>

      <div className="p-2 bg-gray-50 border border-gray-200 rounded-lg">
        <p className="text-xs text-gray-600">
          💡 LoRA model will be activated when generating images
        </p>
      </div>
    </div>
  )
} 