/**
 * Root-level Image Proxy for Cloudflare Pages
 * Alternative route: /image-proxy
 */

export async function onRequest(context) {
  const { request } = context;
  const url = new URL(request.url);
  const targetUrl = url.searchParams.get('url');

  // CORS for all requests
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': '*',
  };

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (!targetUrl) {
    return new Response('No URL provided', { 
      status: 400, 
      headers: { ...corsHeaders, 'Content-Type': 'text/plain' }
    });
  }

  try {
    const response = await fetch(targetUrl);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.status}`);
    }

    const data = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';

    return new Response(data, {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });

  } catch (error) {
    return new Response(`Error: ${error.message}`, {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'text/plain' },
    });
  }
} 