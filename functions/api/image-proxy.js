/**
 * Cloudflare Pages Function: Image Proxy (root-level) with Debug Logging
 * Route: /api/image-proxy
 */

export async function onRequest(context) {
  console.log('=== IMAGE PROXY FUNCTION CALLED ===');
  
  const { request } = context;
  const url = new URL(request.url);
  const targetUrl = url.searchParams.get('url');
  
  console.log('Request URL:', request.url);
  console.log('Request Method:', request.method);
  console.log('Target URL:', targetUrl);
  console.log('Request Headers:', Object.fromEntries(request.headers));

  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Max-Age': '86400',
  };

  // CORS preflight
  if (request.method === 'OPTIONS') {
    console.log('CORS preflight request - returning OPTIONS response');
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (!targetUrl) {
    console.log('ERROR: Missing url parameter');
    return new Response('Missing url parameter', { status: 400, headers: { ...corsHeaders, 'Content-Type': 'text/plain' } });
  }

  try {
    console.log('Starting fetch to target URL:', targetUrl);
    
    const response = await fetch(targetUrl, {
      headers: {
        'Accept': 'image/*',
        'User-Agent': 'CF-Pages-Proxy-Debug',
      },
    });

    console.log('Fetch response status:', response.status);
    console.log('Fetch response headers:', Object.fromEntries(response.headers));

    if (!response.ok) {
      console.log('ERROR: Fetch failed with status:', response.status);
      throw new Error(`HTTP ${response.status}`);
    }

    const imageData = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';
    
    console.log('Successfully fetched image data, size:', imageData.byteLength, 'bytes');
    console.log('Content-Type:', contentType);

    const responseHeaders = {
      ...corsHeaders,
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=3600',
      'X-Proxy-Debug': 'success',
    };
    
    console.log('Returning successful response with headers:', responseHeaders);

    return new Response(imageData, {
      status: 200,
      headers: responseHeaders,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.log('ERROR in proxy function:', message);
    console.log('Error stack:', err.stack);
    
    return new Response(`Proxy failed: ${message}`, { 
      status: 500, 
      headers: { 
        ...corsHeaders, 
        'Content-Type': 'text/plain',
        'X-Proxy-Debug': 'failed',
        'X-Error-Message': message
      } 
    });
  }
} 