/**
 * Cloudflare Pages Function: Image Proxy (root-level)
 * Route: /api/image-proxy
 */

export async function onRequest(context) {
  const { request } = context;
  const url = new URL(request.url);
  const targetUrl = url.searchParams.get('url');

  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Max-Age': '86400',
  };

  // CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (!targetUrl) {
    return new Response('Missing url parameter', { status: 400, headers: { ...corsHeaders, 'Content-Type': 'text/plain' } });
  }

  try {
    const response = await fetch(targetUrl, {
      headers: {
        'Accept': 'image/*',
        'User-Agent': 'CF-Pages-Proxy',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const imageData = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/png';

    return new Response(imageData, {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(`Proxy failed: ${message}`, { status: 500, headers: { ...corsHeaders, 'Content-Type': 'text/plain' } });
  }
} 