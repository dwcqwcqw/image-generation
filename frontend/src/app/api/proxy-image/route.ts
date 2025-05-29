import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const imageUrl = searchParams.get('url');
  
  if (!imageUrl) {
    return NextResponse.json({ error: 'URL parameter required' }, { status: 400 });
  }

  // 验证URL是否来自允许的域名
  try {
    const url = new URL(imageUrl);
    const allowedHosts = [
      'image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com',
      'c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com'
    ];
    
    if (!allowedHosts.includes(url.hostname)) {
      return NextResponse.json({ error: 'Unauthorized domain' }, { status: 403 });
    }

    console.log(`[Image Proxy] Fetching: ${imageUrl}`);
    
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'NextJS-ImageProxy/1.0',
      },
    });
    
    if (!response.ok) {
      console.error(`[Image Proxy] HTTP ${response.status} for ${imageUrl}`);
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const imageBuffer = await response.arrayBuffer();
    const contentType = response.headers.get('Content-Type') || 'image/png';
    
    console.log(`[Image Proxy] Success: ${imageBuffer.byteLength} bytes, type: ${contentType}`);
    
    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Cache-Control': 'public, max-age=3600, s-maxage=3600',
        'CDN-Cache-Control': 'public, max-age=86400',
        'Vercel-CDN-Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('[Image Proxy] Error:', error);
    return NextResponse.json({ 
      error: 'Failed to fetch image', 
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// 处理CORS预检请求
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': '*',
    },
  });
} 