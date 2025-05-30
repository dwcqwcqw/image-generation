import { NextRequest, NextResponse } from 'next/server'
import axios from 'axios'

export async function POST(request: NextRequest) {
  try {
    const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
    const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
      console.error('Missing RunPod configuration')
      
      // For local testing, return a mock response
      const body = await request.json()
      console.log('Local test mode - received request:', body)
      
      return NextResponse.json({
        success: true,
        data: [{
          id: 'test-' + Date.now(),
          url: 'https://via.placeholder.com/512x512/FF6B6B/FFFFFF?text=Test+Image',
          prompt: body.prompt,
          negativePrompt: body.negativePrompt || '',
          seed: body.seed || 12345,
          width: body.width || 512,
          height: body.height || 512,
          steps: body.steps || 20,
          cfgScale: body.cfgScale || 7.0,
          baseModel: body.baseModel || 'realistic',
          createdAt: new Date().toISOString(),
          type: 'text-to-image'
        }]
      })
    }

    const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`
    
    const body = await request.json()
    
    // 验证必需参数
    if (!body.prompt || typeof body.prompt !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Prompt is required' },
        { status: 400 }
      )
    }

    // 准备 RunPod 请求 - 支持新的静态LoRA系统
    const runpodRequest = {
      input: {
        task_type: 'text-to-image',
        prompt: body.prompt,
        negativePrompt: body.negativePrompt || '',
        width: body.width || 512,
        height: body.height || 512,
        steps: body.steps || (body.baseModel === 'realistic' ? 4 : 20),
        cfgScale: body.cfgScale !== undefined ? body.cfgScale : (body.baseModel === 'realistic' ? 0.0 : 7.0),
        seed: body.seed || -1,
        numImages: Math.min(body.numImages || 1, 4), // 限制最多4张
        baseModel: body.baseModel || 'realistic',
        lora_config: body.lora_config || {}
      }
    }

    console.log('Sending request to RunPod:', runpodRequest)

    // 调用 RunPod API
    const response = await axios.post(RUNPOD_API_URL, runpodRequest, {
      headers: {
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 300000, // 5 分钟超时
    })

    console.log('RunPod response status:', response.data.status)

    // 处理不同的响应状态
    if (response.data.status === 'COMPLETED') {
      const output = response.data.output
      
      if (output && output.success) {
        return NextResponse.json({
          success: true,
          data: output.data
        })
      } else {
        return NextResponse.json(
          { success: false, error: output?.error || 'Generation failed' },
          { status: 500 }
        )
      }
    } else if (response.data.status === 'IN_QUEUE' || response.data.status === 'IN_PROGRESS') {
      // 对于队列状态，返回202状态码并让前端处理轮询
      return NextResponse.json(
        { 
          success: false, 
          error: `Job is ${response.data.status.toLowerCase().replace('_', ' ')}. Please try again in a moment.`,
          status: response.data.status,
          jobId: response.data.id
        },
        { status: 202 }
      )
    } else {
      return NextResponse.json(
        { success: false, error: `RunPod job failed with status: ${response.data.status}` },
        { status: 500 }
      )
    }

  } catch (error: any) {
    console.error('Text-to-image generation error:', error)
    
    if (error.code === 'ECONNABORTED') {
      return NextResponse.json(
        { success: false, error: 'Request timeout. Please try again.' },
        { status: 408 }
      )
    }
    
    if (error.response) {
      console.error('RunPod API error response:', error.response.data)
    }
    
    return NextResponse.json(
      { 
        success: false, 
        error: error.response?.data?.error || error.message || 'Internal server error' 
      },
      { status: 500 }
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
} 