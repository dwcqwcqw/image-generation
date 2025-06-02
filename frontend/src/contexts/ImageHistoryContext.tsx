'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import type { GeneratedImage } from '@/types'

interface ImageHistoryContextType {
  textToImageHistory: GeneratedImage[]
  imageToImageHistory: GeneratedImage[]
  addTextToImageHistory: (images: GeneratedImage[]) => void
  addImageToImageHistory: (images: GeneratedImage[]) => void
  clearTextToImageHistory: () => void
  clearImageToImageHistory: () => void
  clearAllHistory: () => void
}

const ImageHistoryContext = createContext<ImageHistoryContextType | undefined>(undefined)

interface ImageHistoryProviderProps {
  children: ReactNode
}

export function ImageHistoryProvider({ children }: ImageHistoryProviderProps) {
  const [textToImageHistory, setTextToImageHistory] = useState<GeneratedImage[]>([])
  const [imageToImageHistory, setImageToImageHistory] = useState<GeneratedImage[]>([])

  // 从localStorage加载历史数据
  useEffect(() => {
    try {
      const savedTextToImageHistory = localStorage.getItem('textToImageHistory')
      if (savedTextToImageHistory) {
        const parsed = JSON.parse(savedTextToImageHistory)
        setTextToImageHistory(parsed)
      }

      const savedImageToImageHistory = localStorage.getItem('imageToImageHistory')
      if (savedImageToImageHistory) {
        const parsed = JSON.parse(savedImageToImageHistory)
        setImageToImageHistory(parsed)
      }
    } catch (error) {
      console.error('Failed to load image history from localStorage:', error)
    }
  }, [])

  // 保存文生图历史
  const addTextToImageHistory = (images: GeneratedImage[]) => {
    setTextToImageHistory(prev => {
      const newHistory = [...images, ...prev]
      // 限制最多保存100张历史图片
      const limited = newHistory.slice(0, 100)
      
      try {
        localStorage.setItem('textToImageHistory', JSON.stringify(limited))
      } catch (error) {
        console.error('Failed to save text-to-image history to localStorage:', error)
      }
      
      return limited
    })
  }

  // 保存图生图历史
  const addImageToImageHistory = (images: GeneratedImage[]) => {
    setImageToImageHistory(prev => {
      const newHistory = [...images, ...prev]
      // 限制最多保存100张历史图片
      const limited = newHistory.slice(0, 100)
      
      try {
        localStorage.setItem('imageToImageHistory', JSON.stringify(limited))
      } catch (error) {
        console.error('Failed to save image-to-image history to localStorage:', error)
      }
      
      return limited
    })
  }

  // 清除文生图历史
  const clearTextToImageHistory = () => {
    setTextToImageHistory([])
    try {
      localStorage.removeItem('textToImageHistory')
    } catch (error) {
      console.error('Failed to clear text-to-image history from localStorage:', error)
    }
  }

  // 清除图生图历史
  const clearImageToImageHistory = () => {
    setImageToImageHistory([])
    try {
      localStorage.removeItem('imageToImageHistory')
    } catch (error) {
      console.error('Failed to clear image-to-image history from localStorage:', error)
    }
  }

  // 清除所有历史
  const clearAllHistory = () => {
    clearTextToImageHistory()
    clearImageToImageHistory()
  }

  const value: ImageHistoryContextType = {
    textToImageHistory,
    imageToImageHistory,
    addTextToImageHistory,
    addImageToImageHistory,
    clearTextToImageHistory,
    clearImageToImageHistory,
    clearAllHistory,
  }

  return (
    <ImageHistoryContext.Provider value={value}>
      {children}
    </ImageHistoryContext.Provider>
  )
}

export function useImageHistory() {
  const context = useContext(ImageHistoryContext)
  if (context === undefined) {
    throw new Error('useImageHistory must be used within a ImageHistoryProvider')
  }
  return context
} 