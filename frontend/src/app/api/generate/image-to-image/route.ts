import { NextRequest, NextResponse } from 'next/server'
import axios from 'axios'

export async function POST(request: NextRequest) {
  try {
    const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
    const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
      console.error('Missing RunPod configuration')
      
      // For local testing, return a mock response
      const formData = await request.formData()
      const prompt = formData.get('prompt') as string
      
      console.log('Local test mode - received image-to-image request:', { prompt })
      
      return NextResponse.json({
        success: true,
        data: [{
          id: 'test-img2img-' + Date.now(),
          url: 'https://via.placeholder.com/512x512/4ECDC4/FFFFFF?text=Image+to+Image+Test',
          prompt: prompt || 'test prompt',
          negativePrompt: formData.get('negativePrompt') as string || '',
          seed: parseInt(formData.get('seed') as string) || 12345,
          width: parseInt(formData.get('width') as string) || 512,
          height: parseInt(formData.get('height') as string) || 512,
          steps: parseInt(formData.get('steps') as string) || 20,
          cfgScale: parseFloat(formData.get('cfgScale') as string) || 7.0,
          createdAt: new Date().toISOString(),
          type: 'image-to-image',
          denoisingStrength: parseFloat(formData.get('denoisingStrength') as string) || 0.8
        }]
      })
    }

    const RUNPOD_API_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`

    const formData = await request.formData()
    const prompt = formData.get('prompt') as string
    const imageFile = formData.get('image') as File

    // 验证必需参数
    if (!prompt || typeof prompt !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Prompt is required' },
        { status: 400 }
      )
    }

    if (!imageFile) {
      return NextResponse.json(
        { success: false, error: 'Image file is required' },
        { status: 400 }
      )
    }

    // 🚨 修复：添加图片大小和格式验证
    const MAX_SIZE = 10 * 1024 * 1024 // 10MB限制
    if (imageFile.size > MAX_SIZE) {
      return NextResponse.json(
        { success: false, error: `图片太大 (${(imageFile.size / 1024 / 1024).toFixed(1)}MB)，请选择小于10MB的图片` },
        { status: 400 }
      )
    }

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if (!validTypes.includes(imageFile.type)) {
      return NextResponse.json(
        { success: false, error: `不支持的图片格式: ${imageFile.type}，请使用JPG、PNG或WebP格式` },
        { status: 400 }
      )
    }

    // 转换图片为 base64
    let base64Image: string
    try {
      const bytes = await imageFile.arrayBuffer()
      const buffer = Buffer.from(bytes)
      base64Image = buffer.toString('base64')
      console.log(`图片转换成功: ${imageFile.name} (${(imageFile.size / 1024).toFixed(1)}KB) -> Base64(${base64Image.length} chars)`)
    } catch (conversionError) {
      console.error('图片转换失败:', conversionError)
      return NextResponse.json(
        { success: false, error: '图片转换失败，请尝试其他格式' },
        { status: 400 }
      )
    }

    // 获取其他参数
    const negativePrompt = formData.get('negativePrompt') as string || ''
    const denoisingStrength = parseFloat(formData.get('denoisingStrength') as string) || 0.7
    const width = parseInt(formData.get('width') as string) || 512
    const height = parseInt(formData.get('height') as string) || 512
    const steps = parseInt(formData.get('steps') as string) || 20
    const cfgScale = parseFloat(formData.get('cfgScale') as string) || 7.0
    const seed = parseInt(formData.get('seed') as string) || -1
    const numImages = Math.min(parseInt(formData.get('numImages') as string) || 1, 4)
    const baseModel = formData.get('baseModel') as string || 'realistic'

    // 🚨 修复：使用与后端handler函数一致的参数结构
    const runpodRequest = {
      input: {
        task_type: 'image-to-image',
        // 🚨 修复：使用扁平结构，不嵌套在params中
        prompt,
        negativePrompt,
        image: base64Image,
        width,
        height,
        steps,
        cfgScale,
        seed,
        numImages,
        denoisingStrength,
        baseModel,
        lora_config: {} // 默认空的LoRA配置
      }
    }

    console.log('发送图生图请求到RunPod:', {
      ...runpodRequest,
      input: {
        ...runpodRequest.input,
        image: `[base64数据，长度: ${base64Image.length}]` // 不打印完整base64
      }
    })

    // 调用 RunPod API
    const response = await axios.post(RUNPOD_API_URL, runpodRequest, {
      headers: {
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 300000, // 5 分钟超时
    })

    console.log('RunPod response status:', response.data.status)

    if (response.data.status === 'COMPLETED') {
      const output = response.data.output
      
      if (output.success) {
        console.log(`图生图成功，生成了 ${output.data.length} 张图片`)
        return NextResponse.json({
          success: true,
          data: output.data
        })
      } else {
        console.error('RunPod generation failed:', output.error)
        return NextResponse.json(
          { success: false, error: output.error || 'Generation failed' },
          { status: 500 }
        )
      }
    } else if (response.data.status === 'IN_QUEUE') {
      console.log('RunPod job queued, should poll for completion')
      return NextResponse.json(
        { success: false, error: 'Job queued - please try direct RunPod mode for better handling' },
        { status: 202 }
      )
    } else {
      console.error('RunPod job failed with status:', response.data.status)
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

    // 🚨 修复：更详细的错误信息
    let errorMessage = 'Internal server error'
    if (error.response?.data?.error) {
      errorMessage = error.response.data.error
    } else if (error.message) {
      errorMessage = error.message
    }
    
    return NextResponse.json(
      { 
        success: false, 
        error: errorMessage
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