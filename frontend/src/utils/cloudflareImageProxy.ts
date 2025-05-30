/**
 * Cloudflare Pages 优化的图片代理工具
 * 专门处理 Cloudflare Pages 部署环境的CORS和API路由问题
 */

// R2域名列表
const R2_DOMAINS = [
  'r2.cloudflarestorage.com',
  'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev',
  'image-generation.c7c141c.r2.cloudflarestorage.com'
]

// 检测是否在Cloudflare Pages环境
function isCloudflarePages(): boolean {
  return typeof window !== 'undefined' && 
         (window.location.hostname.includes('.pages.dev') || 
          process.env.CF_PAGES === '1')
}

// 检测是否有RunPod直连配置
function hasRunPodDirectConfig(): boolean {
  return Boolean(
    process.env.NEXT_PUBLIC_RUNPOD_API_KEY && 
    process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID
  )
}

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
 * Cloudflare Pages优化的图片URL获取
 */
export function getCloudflareImageUrl(originalUrl: string): string {
  console.log('[Image URL] Processing URL:', originalUrl)
  
  // 在Cloudflare Pages环境，优先使用代理（因为CORS仍有问题）
  if (needsProxy(originalUrl) && isCloudflarePages()) {
    console.log('[Image URL] Cloudflare Pages detected, using proxy for R2 URL')
    return `/api/image-proxy?url=${encodeURIComponent(originalUrl)}`
  }
  
  // 开发环境或非R2 URL直接返回
  console.log('[Image URL] Using direct access')
  return originalUrl
}

/**
 * 获取备用代理URL（如果直接访问失败）
 */
export function getProxyImageUrl(originalUrl: string): string {
  if (needsProxy(originalUrl)) {
    // 在Cloudflare Pages环境使用Functions代理
    if (isCloudflarePages()) {
      return `/api/image-proxy?url=${encodeURIComponent(originalUrl)}`
    }
    // 开发环境使用API代理
    return `/api/image-proxy?url=${encodeURIComponent(originalUrl)}`
  }
  return originalUrl
}

/**
 * 获取第二备用代理URL
 */
export function getSecondaryProxyUrl(originalUrl: string): string {
  if (needsProxy(originalUrl)) {
    return `/image-proxy?url=${encodeURIComponent(originalUrl)}`
  }
  return originalUrl
}

/**
 * 增强的图片预加载 - 使用直接URL访问
 */
export function preloadCloudflareImage(originalUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    console.log('[Image Preload] Starting preload for:', originalUrl)
    
    const img = new Image()
    img.crossOrigin = 'anonymous'
    
    img.onload = () => {
      console.log('[Image Preload] Successfully loaded:', originalUrl)
      resolve(originalUrl)
    }
    
    img.onerror = (error) => {
      console.error('[Image Preload] Failed to load:', originalUrl, error)
      reject(new Error(`Failed to load image: ${originalUrl}`))
    }
    
    img.src = originalUrl
  })
}

/**
 * Cloudflare Pages优化的图片下载
 */
export async function downloadCloudflareImage(
  originalUrl: string, 
  filename: string = 'image.png'
): Promise<void> {
  console.log('[Download] Starting download for:', originalUrl)

  // 策略1: 直接下载链接
  try {
    const link = document.createElement('a')
    link.href = originalUrl
    link.download = filename.endsWith('.png') ? filename : `${filename}.png`
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    console.log('[Download] Direct download link created')
    return
  } catch (error) {
    console.log('[Download] Direct download failed:', error)
  }

  // 策略2: Fetch下载（CORS已配置）
  try {
    const response = await fetch(originalUrl, {
      mode: 'cors',
      credentials: 'omit',
      headers: {
        'Accept': 'image/*',
      },
    })
    
    if (response.ok) {
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      
      const link = document.createElement('a')
      link.href = url
      link.download = filename.endsWith('.png') ? filename : `${filename}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      window.URL.revokeObjectURL(url)
      console.log('[Download] Fetch download successful')
      return
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
  } catch (error) {
    console.log('[Download] Fetch download failed:', error)
  }

  // 策略3: 新窗口打开
  console.log('[Download] Falling back to new window')
  window.open(originalUrl, '_blank')
  throw new Error('Download failed, opened in new window')
}

/**
 * 批量下载（Cloudflare Pages优化）
 */
export async function downloadAllCloudflareImages(
  images: Array<{url: string, id: string}>
): Promise<void> {
  console.log(`Starting Cloudflare batch download of ${images.length} images`)
  
  // 在Cloudflare Pages环境中，避免并发下载导致的问题
  const downloadDelay = isCloudflarePages() ? 1000 : 200 // 1秒延迟
  
  for (let i = 0; i < images.length; i++) {
    const image = images[i]
    try {
      await downloadCloudflareImage(image.url, `generated_image_${i + 1}_${image.id}.png`)
      
      // 添加延迟避免并发问题
      if (i < images.length - 1) {
        await new Promise(resolve => setTimeout(resolve, downloadDelay))
      }
    } catch (error) {
      console.error(`Failed to download image ${i + 1}:`, error)
    }
  }
  
  console.log('Cloudflare batch download completed')
}

/**
 * 调试函数：验证图片URL处理
 */
export function debugImageUrl(originalUrl: string): void {
  console.group('[Image URL Debug]')
  console.log('Original URL:', originalUrl)
  console.log('Needs proxy:', needsProxy(originalUrl))
  console.log('Is Cloudflare Pages:', isCloudflarePages())
  console.log('Has RunPod config:', hasRunPodDirectConfig())
  console.log('Final URL:', getCloudflareImageUrl(originalUrl))
  
  // 测试URL可访问性
  if (originalUrl) {
    const testImg = new Image()
    testImg.crossOrigin = 'anonymous'
    testImg.onload = () => console.log('✅ URL accessible via direct load')
    testImg.onerror = () => console.log('❌ URL failed direct load')
    testImg.src = originalUrl
  }
  
  console.groupEnd()
} 