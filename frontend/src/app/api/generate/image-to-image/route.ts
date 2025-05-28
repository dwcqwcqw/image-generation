import { NextRequest, NextResponse } from 'next/server'
import axios from 'axios'

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
  console.error('Missing RunPod configuration')
}

const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = error => reject(error)
  })
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    
    // 获取文件和其他参数
    const imageFile = formData.get('image') as File
    const prompt = formData.get('prompt') as string
    const negativePrompt = formData.get('negativePrompt') as string || ''
    const width = parseInt(formData.get('width') as string) || 512
    const height = parseInt(formData.get('height') as string) || 512
    const steps = parseInt(formData.get('steps') as string) || 20
    const cfgScale = parseFloat(formData.get('cfgScale') as string) || 7.0
    const seed = parseInt(formData.get('seed') as string) || -1
    const numImages = Math.min(parseInt(formData.get('numImages') as string) || 1, 4)
    const denoisingStrength = parseFloat(formData.get('denoisingStrength') as string) || 0.7
    
    // 验证必需参数
    if (!prompt || typeof prompt !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Prompt is required' },
        { status: 400 }
      )
    }

    if (!imageFile || !imageFile.type.startsWith('image/')) {
      return NextResponse.json(
        { success: false, error: 'Valid image file is required' },
        { status: 400 }
      )
    }

    // 将图片转换为 base64
    const imageBase64 = await fileToBase64(imageFile)

    // 准备 RunPod 请求
    const runpodRequest = {
      input: {
        task_type: 'image-to-image',
        params: {
          prompt,
          negativePrompt,
          image: imageBase64,
          width,
          height,
          steps,
          cfgScale,
          seed,
          numImages,
          denoisingStrength,
        }
      }
    }

    console.log('Sending image-to-image request to RunPod')

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
    console.error('Image-to-image generation error:', error)
    
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