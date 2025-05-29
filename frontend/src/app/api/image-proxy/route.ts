import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const imageUrl = searchParams.get('url')
    
    if (!imageUrl) {
      return NextResponse.json({ error: 'Missing image URL' }, { status: 400 })
    }

    console.log('Proxying image request:', imageUrl)

    // 验证URL是否来自我们的R2存储
    const allowedDomains = [
      'r2.cloudflarestorage.com',
      'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev',
      'image-generation.c7c141c.r2.cloudflarestorage.com'
    ]

    const urlObj = new URL(imageUrl)
    const isAllowed = allowedDomains.some(domain => 
      urlObj.hostname.includes(domain)
    )

    if (!isAllowed) {
      return NextResponse.json({ error: 'Invalid image URL' }, { status: 403 })
    }

    // 获取图片数据
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'AI-Image-Generation-Frontend/1.0',
      },
    })

    if (!response.ok) {
      console.error('Failed to fetch image:', response.status, response.statusText)
      return NextResponse.json(
        { error: `Failed to fetch image: ${response.status}` }, 
        { status: response.status }
      )
    }

    const imageBuffer = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/png'

    console.log(`Successfully proxied image: ${imageBuffer.byteLength} bytes, type: ${contentType}`)

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
      },
    })

  } catch (error) {
    console.error('Image proxy error:', error)
    return NextResponse.json(
      { error: 'Internal server error' }, 
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