'use client'

import { useState, useEffect } from 'react'
import { toast } from 'react-hot-toast'
import { Wand2, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react'
import { getAvailableLoras, switchLoraModel } from '@/services/api'
import type { LoRAModel } from '@/types'

interface LoRASelectorProps {
  value: string
  onChange: (loraId: string) => void
  disabled?: boolean
}

export default function LoRASelector({ value, onChange, disabled }: LoRASelectorProps) {
  const [loras, setLoras] = useState<Record<string, LoRAModel>>({})
  const [loading, setLoading] = useState(true)
  const [switching, setSwitching] = useState(false)
  const [currentLora, setCurrentLora] = useState(value)

  const loadLoras = async () => {
    try {
      setLoading(true)
      const response = await getAvailableLoras()
      setLoras(response.loras)
      setCurrentLora(response.current)
      if (value !== response.current) {
        onChange(response.current)
      }
    } catch (error: any) {
      console.error('Failed to load LoRA models:', error)
      toast.error('Failed to load LoRA models')
    } finally {
      setLoading(false)
    }
  }

  const handleLoraChange = async (loraId: string) => {
    if (loraId === currentLora || switching || disabled) return

    try {
      setSwitching(true)
      await switchLoraModel(loraId)
      setCurrentLora(loraId)
      onChange(loraId)
      toast.success(`Switched to ${loras[loraId]?.name}`)
    } catch (error: any) {
      console.error('Failed to switch LoRA:', error)
      toast.error(`Failed to switch LoRA: ${error.message}`)
    } finally {
      setSwitching(false)
    }
  }

  useEffect(() => {
    loadLoras()
  }, [])

  if (loading) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Model
        </label>
        <div className="p-3 bg-gray-50 rounded-lg flex items-center space-x-2">
          <RefreshCw className="w-4 h-4 animate-spin text-gray-500" />
          <span className="text-sm text-gray-600">Loading LoRA models...</span>
        </div>
      </div>
    )
  }

  const loraEntries = Object.entries(loras)

  if (loraEntries.length === 0) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Model
        </label>
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-500" />
          <span className="text-sm text-red-600">No LoRA models available</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          LoRA Model
        </label>
        <button
          onClick={loadLoras}
          disabled={loading || switching}
          className="text-xs text-gray-500 hover:text-gray-700 flex items-center space-x-1"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="space-y-2">
        {loraEntries.map(([loraId, lora]) => (
          <button
            key={loraId}
            onClick={() => handleLoraChange(loraId)}
            disabled={disabled || switching}
            className={`w-full p-3 text-left rounded-lg border transition-all duration-200 ${
              currentLora === loraId
                ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
                : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            } ${
              disabled || switching ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <Wand2 className="w-4 h-4 text-primary-600" />
                  <span className="font-medium text-gray-900">{lora.name}</span>
                  {currentLora === loraId && (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  )}
                </div>
                <p className="text-xs text-gray-600 mt-1">{lora.description}</p>
              </div>
              
              {switching && currentLora !== loraId && (
                <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />
              )}
            </div>
          </button>
        ))}
      </div>

      {switching && (
        <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
            <span className="text-sm text-blue-700">Switching LoRA model...</span>
          </div>
        </div>
      )}
    </div>
  )
} 