/**
 * 图片代理工具 - 解决CORS和图片显示问题的根本方案
 */

// R2域名列表
const R2_DOMAINS = [
  'r2.cloudflarestorage.com',
  'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev',
  'image-generation.c7c141c.r2.cloudflarestorage.com'
]

/**
 * 检查URL是否需要代理
 */
export function needsProxy(url: string): boolean {
  try {
    const urlObj = new URL(url)
    return R2_DOMAINS.some(domain => urlObj.hostname.includes(domain))
  } catch {
    return false
  }
}

/**
 * 将R2 URL转换为代理URL
 */
export function getProxiedImageUrl(originalUrl: string): string {
  if (!needsProxy(originalUrl)) {
    return originalUrl
  }

  // 使用我们的前端API代理
  const proxyUrl = `/api/image-proxy?url=${encodeURIComponent(originalUrl)}`
  
  console.log('Converting to proxy URL:', {
    original: originalUrl,
    proxied: proxyUrl
  })
  
  return proxyUrl
}

/**
 * 下载图片 - 使用多重策略确保成功
 */
export async function downloadImage(
  originalUrl: string, 
  filename: string = 'image.png'
): Promise<void> {
  try {
    console.log('Starting image download:', originalUrl)

    // 策略1: 尝试直接下载
    let downloadUrl = originalUrl
    let response: Response

    try {
      response = await fetch(originalUrl)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
    } catch (error) {
      console.log('Direct download failed, using proxy:', error)
      
      // 策略2: 使用代理下载
      downloadUrl = getProxiedImageUrl(originalUrl)
      response = await fetch(downloadUrl)
      
      if (!response.ok) {
        throw new Error(`Proxy download failed: HTTP ${response.status}`)
      }
    }

    // 获取图片数据
    const blob = await response.blob()
    
    // 策略3: 创建下载链接
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename.endsWith('.png') ? filename : `${filename}.png`
    
    // 触发下载
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // 清理
    window.URL.revokeObjectURL(url)
    
    console.log('Image downloaded successfully:', filename)
    
  } catch (error) {
    console.error('All download strategies failed:', error)
    
    // 策略4: 回退到新窗口打开
    console.log('Falling back to new window opening')
    const proxyUrl = getProxiedImageUrl(originalUrl)
    window.open(proxyUrl, '_blank')
    
    throw new Error('Download failed, opened in new window')
  }
}

/**
 * 预加载图片 - 用于图片库显示
 */
export function preloadImage(originalUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const proxyUrl = getProxiedImageUrl(originalUrl)
    
    img.onload = () => {
      console.log('Image preloaded successfully:', proxyUrl)
      resolve(proxyUrl)
    }
    
    img.onerror = (error) => {
      console.error('Image preload failed:', error)
      reject(new Error('Failed to load image'))
    }
    
    img.src = proxyUrl
  })
}

/**
 * 批量下载图片
 */
export async function downloadAllImages(images: Array<{url: string, id: string}>): Promise<void> {
  console.log(`Starting batch download of ${images.length} images`)
  
  const downloadPromises = images.map((image, index) => 
    downloadImage(image.url, `generated_image_${index + 1}_${image.id}.png`)
      .catch(error => {
        console.error(`Failed to download image ${index + 1}:`, error)
        return null // 继续其他下载
      })
  )
  
  await Promise.allSettled(downloadPromises)
  console.log('Batch download completed')
} 