/**
 * Cloudflare Pages Function for Image Proxy
 * 专门为Cloudflare Pages环境设计的图片代理API
 */

interface Env {
  // Cloudflare Pages环境变量
}

export async function onRequestGet(context: any): Promise<Response> {
  try {
    const { request } = context
    const url = new URL(request.url)
    const imageUrl = url.searchParams.get('url')
    
    if (!imageUrl) {
      return new Response(JSON.stringify({ error: 'Missing image URL' }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    console.log('Cloudflare Pages Image Proxy:', imageUrl)

    // 验证URL域名
    const allowedDomains = [
      'r2.cloudflarestorage.com',
      'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev',
      'image-generation.c7c141c',
      'cloudflarestorage.com'
    ]

    let isAllowed = false
    try {
      const urlObj = new URL(imageUrl)
      isAllowed = allowedDomains.some(domain => 
        urlObj.hostname.includes(domain)
      )
    } catch (error) {
      return new Response(JSON.stringify({ error: 'Invalid image URL format' }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    if (!isAllowed) {
      return new Response(JSON.stringify({ error: 'Invalid image URL domain' }), {
        status: 403,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    // 获取图片数据
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'CF-Pages-Image-Proxy/1.0',
        'Accept': 'image/*',
      },
    })

    if (!response.ok) {
      return new Response(JSON.stringify({ 
        error: `Failed to fetch image: HTTP ${response.status}` 
      }), {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    const imageData = await response.arrayBuffer()
    const contentType = response.headers.get('content-type') || 'image/png'

    // 返回图片数据
    return new Response(imageData, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': imageData.byteLength.toString(),
        'Cache-Control': 'public, max-age=86400', // 24小时缓存
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'CF-Cache-Status': 'HIT',
      },
    })

  } catch (error) {
    console.error('Cloudflare Pages Image Proxy Error:', error)
    return new Response(JSON.stringify({ 
      error: 'Internal server error', 
      details: String(error) 
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    })
  }
}

// 处理OPTIONS请求 (CORS preflight)
export async function onRequestOptions(): Promise<Response> {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
} 