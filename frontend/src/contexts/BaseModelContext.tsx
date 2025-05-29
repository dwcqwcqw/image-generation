'use client'

import { createContext, useContext, useState } from 'react'
import type { BaseModelType, LoRAConfig } from '@/types'

// Create context for base model state
interface BaseModelContextType {
  baseModel: BaseModelType
  setBaseModel: (model: BaseModelType) => void
  loraConfig: LoRAConfig
  setLoraConfig: (config: LoRAConfig) => void
}

const BaseModelContext = createContext<BaseModelContextType | undefined>(undefined)

export const useBaseModel = () => {
  const context = useContext(BaseModelContext)
  if (!context) {
    throw new Error('useBaseModel must be used within a BaseModelProvider')
  }
  return context
}

export const BaseModelProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [baseModel, setBaseModel] = useState<BaseModelType>('realistic')
  const [loraConfig, setLoraConfig] = useState<LoRAConfig>({ flux_nsfw: 1.0 })

  const handleSetBaseModel = (newBaseModel: BaseModelType) => {
    setBaseModel(newBaseModel)
    // Auto-configure corresponding LoRA
    const newLoRAConfig: LoRAConfig = newBaseModel === 'realistic' 
      ? { flux_nsfw: 1.0 } 
      : { gayporn: 1.0 }
    setLoraConfig(newLoRAConfig)
  }

  return (
    <BaseModelContext.Provider value={{
      baseModel,
      setBaseModel: handleSetBaseModel, // Use the new handler
      loraConfig,
      setLoraConfig
    }}>
      {children}
    </BaseModelContext.Provider>
  )
} 