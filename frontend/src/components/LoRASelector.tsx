'use client'

import { Fragment, useState, useEffect } from 'react'
import { Listbox, Transition } from '@headlessui/react'
import { ChevronsUpDown, Check } from 'lucide-react'
import type { LoRAConfig, BaseModelType } from '@/types'

interface LoRAOption {
  id: string
  name: string
  description: string
}

interface LoRASelectorProps {
  value: LoRAConfig
  onChange: (config: LoRAConfig) => void
  baseModel: BaseModelType
  disabled?: boolean
}

// 静态LoRA列表 - 前端直接显示，后端动态搜索
const STATIC_LORAS = {
  realistic: [
    { id: 'flux_nsfw', name: 'FLUX NSFW', description: 'NSFW真人内容生成模型' },
    { id: 'chastity_cage', name: 'Chastity Cage', description: '贞操笼主题内容生成' },
    { id: 'dynamic_penis', name: 'Dynamic Penis', description: '动态男性解剖生成' },
    { id: 'masturbation', name: 'Masturbation', description: '自慰主题内容生成' },
    { id: 'puppy_mask', name: 'Puppy Mask', description: '小狗面具主题内容' },
    { id: 'butt_and_feet', name: 'Butt and Feet', description: '臀部和足部主题内容' },
    { id: 'cumshots', name: 'Cumshots', description: '射精主题内容生成' },
    { id: 'uncutpenis', name: 'Uncut Penis', description: '未割包皮主题内容' },
    { id: 'doggystyle', name: 'Doggystyle', description: '后入式主题内容' },
    { id: 'fisting', name: 'Fisting', description: '拳交主题内容生成' },
    { id: 'on_off', name: 'On Off', description: '穿衣/脱衣对比内容' },
    { id: 'blowjob', name: 'Blowjob', description: '口交主题内容生成' },
    { id: 'cum_on_face', name: 'Cum on Face', description: '颜射主题内容生成' }
  ],
  anime: [
    { id: 'gayporn', name: 'Gayporn', description: '男同动漫风格内容生成' }
  ]
}

export default function LoRASelector({ value, onChange, baseModel, disabled = false }: LoRASelectorProps) {
  const [selectedLoRA, setSelectedLoRA] = useState<LoRAOption | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // 获取当前基础模型的LoRA列表
  const availableLoras = STATIC_LORAS[baseModel] || []
  
  // 默认选择第一个LoRA
  useEffect(() => {
    if (availableLoras.length > 0 && !selectedLoRA) {
      const defaultLora = availableLoras[0] // 默认选择第一个（FLUX NSFW）
      setSelectedLoRA(defaultLora)
      onChange({ [defaultLora.id]: 1.0 })
    }
  }, [baseModel, availableLoras, selectedLoRA, onChange])

  // 当基础模型改变时重置选择
  useEffect(() => {
    if (availableLoras.length > 0) {
      const defaultLora = availableLoras[0]
      setSelectedLoRA(defaultLora)
      onChange({ [defaultLora.id]: 1.0 })
    }
  }, [baseModel])

  const handleLoRAChange = async (lora: LoRAOption) => {
    if (disabled || isLoading) return

    setIsLoading(true)
    try {
      // 调用后端切换LoRA
      const response = await fetch('/api/loras/switch-single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lora_id: lora.id
        }),
      })

      if (response.ok) {
        setSelectedLoRA(lora)
        onChange({ [lora.id]: 1.0 })
      } else {
        console.error('Failed to switch LoRA:', await response.text())
      }
    } catch (error) {
      console.error('Error switching LoRA:', error)
    } finally {
      setIsLoading(false)
    }
  }

  if (availableLoras.length === 0) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          LoRA模型
        </label>
        <div className="text-sm text-gray-500">
          当前基础模型暂无可用LoRA
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        LoRA模型 ({baseModel === 'realistic' ? '真人风格' : '动漫风格'})
      </label>
      
      <Listbox value={selectedLoRA} onChange={handleLoRAChange} disabled={disabled || isLoading}>
        <div className="relative">
          <Listbox.Button className="relative w-full cursor-default rounded-lg bg-white py-2 pl-3 pr-10 text-left shadow-md focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 sm:text-sm border border-gray-300">
            <span className="block truncate">
              {isLoading ? '切换中...' : selectedLoRA?.name || '选择LoRA模型'}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronsUpDown
                className="h-5 w-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </Listbox.Button>
          
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options className="absolute z-10 mt-1 max-h-32 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
              {availableLoras.map((lora) => (
                <Listbox.Option
                  key={lora.id}
                  className={({ active }) =>
                    `relative cursor-default select-none py-2 pl-10 pr-4 ${
                      active ? 'bg-amber-100 text-amber-900' : 'text-gray-900'
                    }`
                  }
                  value={lora}
                >
                  {({ selected }) => (
                    <>
                      <span
                        className={`block truncate ${
                          selected ? 'font-medium' : 'font-normal'
                        }`}
                      >
                        {lora.name}
                      </span>
                      {selected ? (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-amber-600">
                          <Check className="h-5 w-5" aria-hidden="true" />
                        </span>
                      ) : null}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>
      
      {selectedLoRA && (
        <p className="text-xs text-gray-500 mt-1">
          {selectedLoRA.description}
        </p>
      )}
    </div>
  )
} 