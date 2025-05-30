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
  if (!needsProxy(originalUrl)) {
    return originalUrl
  }

  // 策略1: 如果在Cloudflare Pages环境且有直连配置，尝试添加CORS头
  if (isCloudflarePages()) {
    try {
      const urlObj = new URL(originalUrl)
      // 添加Cloudflare的CORS参数（如果R2支持）
      urlObj.searchParams.set('cf-cache', '1')
      return urlObj.toString()
    } catch {
      // 如果URL解析失败，继续原来的逻辑
    }
  }

  // 策略2: 使用API代理（仅在开发环境或支持的环境）
  if (!isCloudflarePages()) {
    const proxyUrl = `/api/image-proxy?url=${encodeURIComponent(originalUrl)}`
    console.log('Using API proxy for:', originalUrl)
    return proxyUrl
  }

  // 策略3: Cloudflare Pages环境直接返回原URL（可能需要其他配置）
  console.log('Cloudflare Pages: Using direct URL:', originalUrl)
  return originalUrl
}

/**
 * 增强的图片预加载 - 支持多种加载策略
 */
export function preloadCloudflareImage(originalUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    
    // 添加CORS属性
    img.crossOrigin = 'anonymous'
    
    const tryUrls = [
      getCloudflareImageUrl(originalUrl),
      originalUrl,
      // 备用URL策略
      originalUrl.replace('r2.cloudflarestorage.com', 'pub-cb95af834c6b4d0d9b55f72e0f5e7d3d.r2.dev')
    ]
    
    let currentIndex = 0
    
    function tryNextUrl() {
      if (currentIndex >= tryUrls.length) {
        reject(new Error('All image loading strategies failed'))
        return
      }
      
      const currentUrl = tryUrls[currentIndex]
      console.log(`Trying image URL ${currentIndex + 1}:`, currentUrl)
      
      img.onload = () => {
        console.log('Image loaded successfully:', currentUrl)
        resolve(currentUrl)
      }
      
      img.onerror = () => {
        console.log('Image load failed:', currentUrl)
        currentIndex++
        tryNextUrl()
      }
      
      img.src = currentUrl
    }
    
    tryNextUrl()
  })
}

/**
 * Cloudflare Pages优化的图片下载
 */
export async function downloadCloudflareImage(
  originalUrl: string, 
  filename: string = 'image.png'
): Promise<void> {
  console.log('Starting Cloudflare-optimized download:', originalUrl)

  // 策略1: 直接下载链接
  try {
    const link = document.createElement('a')
    link.href = getCloudflareImageUrl(originalUrl)
    link.download = filename.endsWith('.png') ? filename : `${filename}.png`
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    console.log('Direct download initiated')
    return
  } catch (error) {
    console.log('Direct download failed:', error)
  }

  // 策略2: Fetch下载（如果支持CORS）
  try {
    const response = await fetch(getCloudflareImageUrl(originalUrl), {
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
      console.log('Fetch download successful')
      return
    }
  } catch (error) {
    console.log('Fetch download failed:', error)
  }

  // 策略3: 新窗口打开
  console.log('Falling back to new window')
  window.open(getCloudflareImageUrl(originalUrl), '_blank')
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