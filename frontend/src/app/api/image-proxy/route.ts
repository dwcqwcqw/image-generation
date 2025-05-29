import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const imageUrl = searchParams.get('url')
    
    if (!imageUrl) {
      console.error('Image proxy: Missing image URL')
      return NextResponse.json({ error: 'Missing image URL' }, { status: 400 })
    }

    console.log('Image proxy: Requesting URL:', imageUrl)

    // 验证URL是否来自我们的R2存储 - 更宽松的验证
    const allowedDomains = [
      'r2.cloudflarestorage.com',
      'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev',
      'image-generation.c7c141c',  // 部分匹配
      'cloudflarestorage.com'      // 宽松匹配
    ]

    let isAllowed = false
    try {
      const urlObj = new URL(imageUrl)
      isAllowed = allowedDomains.some(domain => 
        urlObj.hostname.includes(domain)
      )
      console.log('Image proxy: URL hostname:', urlObj.hostname, 'isAllowed:', isAllowed)
    } catch (error) {
      console.error('Image proxy: Invalid URL:', error)
      return NextResponse.json({ error: 'Invalid image URL format' }, { status: 400 })
    }

    if (!isAllowed) {
      console.error('Image proxy: Domain not allowed:', imageUrl)
      return NextResponse.json({ error: 'Invalid image URL domain' }, { status: 403 })
    }

    // 获取图片数据 - 增加超时和错误处理
    let response: Response
    try {
      console.log('Image proxy: Fetching image...')
      response = await fetch(imageUrl, {
        headers: {
          'User-Agent': 'AI-Image-Generation-Frontend/1.0',
          'Accept': 'image/*',
        },
        // 增加超时
        signal: AbortSignal.timeout(30000), // 30秒超时
      })
      
      console.log('Image proxy: Fetch response:', response.status, response.statusText)
    } catch (error) {
      console.error('Image proxy: Fetch failed:', error)
      return NextResponse.json(
        { error: `Failed to fetch image: ${error}` }, 
        { status: 502 }
      )
    }

    if (!response.ok) {
      console.error('Image proxy: HTTP error:', response.status, response.statusText)
      return NextResponse.json(
        { error: `Failed to fetch image: HTTP ${response.status}` }, 
        { status: response.status }
      )
    }

    let imageBuffer: ArrayBuffer
    let contentType: string
    
    try {
      imageBuffer = await response.arrayBuffer()
      contentType = response.headers.get('content-type') || 'image/png'
      
      console.log(`Image proxy: Success - ${imageBuffer.byteLength} bytes, type: ${contentType}`)
    } catch (error) {
      console.error('Image proxy: Failed to read response:', error)
      return NextResponse.json(
        { error: 'Failed to read image data' }, 
        { status: 502 }
      )
    }

    // 验证图片数据
    if (imageBuffer.byteLength === 0) {
      console.error('Image proxy: Empty image data')
      return NextResponse.json(
        { error: 'Empty image data' }, 
        { status: 502 }
      )
    }

    // 返回图片数据，设置正确的CORS头
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': imageBuffer.byteLength.toString(),
        'Cache-Control': 'public, max-age=86400', // 24小时缓存
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'X-Proxy-Success': 'true',
      },
    })

  } catch (error) {
    console.error('Image proxy: Unexpected error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: String(error) }, 
      { status: 500 }
    )
  }
}

// 处理OPTIONS请求 (CORS preflight)
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
} 