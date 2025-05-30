/**
 * Cloudflare Pages Function: Image Proxy with CORS Support
 * Handles CORS issues when accessing R2 storage from Pages
 */

export async function onRequest(context: any) {
  const { request } = context;
  const url = new URL(request.url);
  const targetUrl = url.searchParams.get('url');

  // CORS preflight handling
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Max-Age': '86400',
      },
    });
  }

  if (!targetUrl) {
    return new Response('Missing url parameter', { 
      status: 400,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'text/plain',
      }
    });
  }

  try {
    console.log('[Image Proxy] Proxying request to:', targetUrl);
    
    // Fetch the image from R2
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: {
        'Accept': 'image/*',
        'User-Agent': 'Cloudflare-Pages-Function',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    // Get the image data
    const imageData = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';

    console.log('[Image Proxy] Successfully proxied image, size:', imageData.byteLength);

    // Return the image with CORS headers
    return new Response(imageData, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Content-Length': imageData.byteLength.toString(),
      },
    });

  } catch (error) {
    console.error('[Image Proxy] Error:', error);
    
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    return new Response(`Proxy error: ${errorMessage}`, {
      status: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'text/plain',
      },
    });
  }
} 