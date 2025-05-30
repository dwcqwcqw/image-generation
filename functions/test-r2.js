/**
 * Test R2 URL accessibility - Route: /test-r2
 */

export async function onRequest(context) {
  console.log('=== R2 TEST FUNCTION CALLED ===');
  
  const { request } = context;
  const url = new URL(request.url);
  const testUrl = url.searchParams.get('url');
  
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': '*',
  };

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (!testUrl) {
    return new Response('Missing url parameter', { 
      status: 400, 
      headers: { ...corsHeaders, 'Content-Type': 'text/plain' } 
    });
  }

  console.log('Testing URL:', testUrl);

  try {
    // Test HEAD request first (faster)
    console.log('Testing HEAD request...');
    const headResponse = await fetch(testUrl, { method: 'HEAD' });
    
    console.log('HEAD response status:', headResponse.status);
    console.log('HEAD response headers:', Object.fromEntries(headResponse.headers));
    
    const result = {
      url: testUrl,
      headRequest: {
        status: headResponse.status,
        statusText: headResponse.statusText,
        headers: Object.fromEntries(headResponse.headers),
        ok: headResponse.ok
      }
    };

    // If HEAD fails, try GET for more info
    if (!headResponse.ok) {
      console.log('HEAD failed, trying GET...');
      try {
        const getResponse = await fetch(testUrl, { method: 'GET' });
        
        let responseBody = '';
        try {
          responseBody = await getResponse.text();
        } catch (e) {
          responseBody = 'Could not read response body';
        }
        
        result.getRequest = {
          status: getResponse.status,
          statusText: getResponse.statusText,
          headers: Object.fromEntries(getResponse.headers),
          body: responseBody.substring(0, 500),
          ok: getResponse.ok
        };
      } catch (getError) {
        result.getRequest = { error: getError.message };
      }
    }

    return new Response(JSON.stringify(result, null, 2), {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json',
      },
    });

  } catch (error) {
    console.log('Test error:', error);
    
    const errorResult = {
      url: testUrl,
      error: error.message,
      stack: error.stack
    };

    return new Response(JSON.stringify(errorResult, null, 2), {
      status: 500,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json',
      },
    });
  }
} 