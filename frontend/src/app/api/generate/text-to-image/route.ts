import { NextRequest, NextResponse } from 'next/server'
import axios from 'axios'

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
  console.error('Missing RunPod configuration')
}

const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // 验证必需参数
    if (!body.prompt || typeof body.prompt !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Prompt is required' },
        { status: 400 }
      )
    }

    // 准备 RunPod 请求
    const runpodRequest = {
      input: {
        task_type: 'text-to-image',
        params: {
          prompt: body.prompt,
          negativePrompt: body.negativePrompt || '',
          width: body.width || 512,
          height: body.height || 512,
          steps: body.steps || 20,
          cfgScale: body.cfgScale || 7.0,
          seed: body.seed || -1,
          numImages: Math.min(body.numImages || 1, 4), // 限制最多4张
        }
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

    console.log('RunPod response:', response.data)

    if (response.data.status === 'COMPLETED') {
      const output = response.data.output
      
      if (output.success) {
        return NextResponse.json({
          success: true,
          data: output.data
        })
      } else {
        return NextResponse.json(
          { success: false, error: output.error || 'Generation failed' },
          { status: 500 }
        )
      }
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