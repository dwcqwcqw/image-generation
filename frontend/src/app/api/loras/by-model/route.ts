import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${process.env.RUNPOD_ENDPOINT_URL}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({
        input: {
          task_type: 'get-loras-by-model'
        }
      }),
    })

    if (!response.ok) {
      throw new Error(`RunPod API error: ${response.status}`)
    }

    const data = await response.json()
    
    if (data.error) {
      throw new Error(data.error)
    }

    return NextResponse.json(data.output?.data || {})
  } catch (error) {
    console.error('Error fetching LoRAs by model:', error)
    return NextResponse.json(
      { error: 'Failed to fetch LoRAs by model' },
      { status: 500 }
    )
  }
} 