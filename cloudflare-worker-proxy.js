// Cloudflare Worker for Image Proxy
// Deploy this to Cloudflare Workers for proper CORS handling

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // Only handle GET requests
  if (request.method !== 'GET') {
    return new Response('Method not allowed', { status: 405 })
  }
  
  // Get the image URL from query parameters
  const imageUrl = url.searchParams.get('url')
  
  if (!imageUrl) {
    return new Response('Missing url parameter', { status: 400 })
  }
  
  try {
    // Validate the URL is from allowed domains
    const targetUrl = new URL(imageUrl)
    const allowedHosts = [
      'image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com',
      'c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com'
    ]
    
    if (!allowedHosts.includes(targetUrl.hostname)) {
      return new Response('Unauthorized domain', { status: 403 })
    }
    
    // Fetch the image
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'CloudflareWorker-ImageProxy/1.0'
      }
    })
    
    if (!response.ok) {
      return new Response(`Failed to fetch image: ${response.status}`, { status: response.status })
    }
    
    // Create new response with CORS headers
    const newResponse = new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'image/png',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Cache-Control': 'public, max-age=3600',
        'CDN-Cache-Control': 'public, max-age=86400'
      }
    })
    
    return newResponse
    
  } catch (error) {
    return new Response(`Error: ${error.message}`, { status: 500 })
  }
}

// Handle CORS preflight requests
addEventListener('fetch', event => {
  if (event.request.method === 'OPTIONS') {
    event.respondWith(new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': '*'
      }
    }))
  }
}) 