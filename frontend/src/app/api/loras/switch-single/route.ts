import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { lora_id } = body

    if (!lora_id) {
      return NextResponse.json(
        { error: 'lora_id is required' },
        { status: 400 }
      )
    }

    const response = await fetch(`${process.env.RUNPOD_ENDPOINT_URL}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({
        input: {
          task_type: 'switch-single-lora',
          lora_id: lora_id
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
    console.error('Error switching single LoRA:', error)
    return NextResponse.json(
      { error: 'Failed to switch LoRA' },
      { status: 500 }
    )
  }
} 