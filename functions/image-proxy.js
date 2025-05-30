/**
 * Cloudflare Pages Function: Image Proxy (no prefix)
 * Route: /image-proxy
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

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (!targetUrl) {
    return new Response('No URL provided', { status: 400, headers: { ...corsHeaders, 'Content-Type': 'text/plain' } });
  }

  try {
    const res = await fetch(targetUrl);
    if (!res.ok) throw new Error(`Upstream ${res.status}`);
    const data = await res.arrayBuffer();
    const contentType = res.headers.get('content-type') || 'image/png';

    return new Response(data, { status: 200, headers: { ...corsHeaders, 'Content-Type': contentType, 'Cache-Control': 'public, max-age=3600' } });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(`Error: ${message}`, { status: 500, headers: { ...corsHeaders, 'Content-Type': 'text/plain' } });
  }
} 