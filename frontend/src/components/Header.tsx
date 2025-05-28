'use client'

import { Sparkles } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AI Image Generator</h1>
              <p className="text-sm text-gray-500">Powered by FLUX</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center space-x-6">
            <a 
              href="#" 
              className="text-gray-600 hover:text-gray-900 transition-colors duration-200"
            >
              Gallery
            </a>
            <a 
              href="#" 
              className="text-gray-600 hover:text-gray-900 transition-colors duration-200"
            >
              API
            </a>
            <a 
              href="#" 
              className="text-gray-600 hover:text-gray-900 transition-colors duration-200"
            >
              Pricing
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
} 