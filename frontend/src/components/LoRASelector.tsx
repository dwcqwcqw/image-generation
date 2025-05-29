'use client'

import { Switch } from '@headlessui/react'
import type { LoRAConfig, BaseModelType } from '@/types'

interface LoRASelectorProps {
  value: LoRAConfig
  onChange: (config: LoRAConfig) => void
  baseModel: BaseModelType
  disabled?: boolean
}

export default function LoRASelector({ value, onChange, baseModel, disabled = false }: LoRASelectorProps) {
  // Get the active LoRA for the current base model
  const getActiveLoRA = () => {
    if (baseModel === 'realistic') {
      return {
        id: 'flux_nsfw',
        name: 'FLUX NSFW',
        description: 'Enhanced realistic human content generation'
      }
    } else {
      return {
        id: 'gayporn',
        name: 'Gayporn',
        description: 'Specialized anime-style content generation'
      }
    }
  }

  const activeLoRA = getActiveLoRA()
  const isEnabled = (value[activeLoRA.id as keyof LoRAConfig] || 0) > 0

  const handleToggle = (enabled: boolean) => {
    const newConfig: LoRAConfig = {
      ...value,
      [activeLoRA.id]: enabled ? 1.0 : 0
    }
    onChange(newConfig)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Enhancement
        </label>
      </div>

      {/* LoRA Toggle Card */}
      <div className={`p-4 rounded-lg border-2 transition-all duration-200 ${
        isEnabled
          ? 'border-primary-500 bg-primary-50'
          : 'border-gray-200 bg-white'
      } ${disabled ? 'opacity-50' : ''}`}>
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                isEnabled ? 'bg-primary-100' : 'bg-gray-100'
              }`}>
                <span className={`text-lg font-bold ${
                  isEnabled ? 'text-primary-600' : 'text-gray-600'
                }`}>
                  ✨
                </span>
              </div>
              <div>
                <h3 className={`font-medium ${
                  isEnabled ? 'text-primary-900' : 'text-gray-900'
                }`}>
                  {activeLoRA.name}
                </h3>
                <p className={`text-sm ${
                  isEnabled ? 'text-primary-700' : 'text-gray-600'
                }`}>
                  {activeLoRA.description}
                </p>
              </div>
            </div>
          </div>
          
          <Switch
            checked={isEnabled}
            onChange={handleToggle}
            disabled={disabled}
            className={`${
              isEnabled ? 'bg-primary-600' : 'bg-gray-200'
            } relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
              disabled ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            <span className="sr-only">Toggle LoRA enhancement</span>
            <span
              aria-hidden="true"
              className={`${
                isEnabled ? 'translate-x-5' : 'translate-x-0'
              } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
            />
          </Switch>
        </div>
      </div>

      {/* Status Info */}
      <div className={`p-3 rounded-lg border ${
        isEnabled 
          ? 'bg-green-50 border-green-200' 
          : 'bg-gray-50 border-gray-200'
      }`}>
        <div className="flex items-start space-x-2">
          <div className={`w-4 h-4 rounded-full flex-shrink-0 mt-0.5 ${
            isEnabled ? 'bg-green-500' : 'bg-gray-400'
          }`}></div>
          <div className="text-xs">
            <p className={`font-medium ${
              isEnabled ? 'text-green-800' : 'text-gray-700'
            }`}>
              {isEnabled ? 'LoRA Enhanced' : 'Standard Generation'}
            </p>
            <p className={`mt-1 ${
              isEnabled ? 'text-green-700' : 'text-gray-600'
            }`}>
              {isEnabled 
                ? `Using ${activeLoRA.name} enhancement for improved quality and style consistency.`
                : 'Using base model only. Enable LoRA for enhanced results.'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 