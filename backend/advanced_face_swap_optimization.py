#!/usr/bin/env python3
"""
高级换脸优化模块 - 解决CUDA兼容性、动态混合、高精度检测等问题
Advanced Face Swap Optimization Module
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFaceSwapOptimizer:
    """高级换脸优化器"""
    
    def __init__(self):
        self.face_analyser = None
        self.face_swapper = None
        self.face_enhancer = None
        self.execution_providers = self._get_optimized_execution_providers()
        
    def _get_optimized_execution_providers(self) -> List[Any]:
        """获取优化的执行提供程序，解决CUDA兼容性问题"""
        providers = []
        
        if torch.cuda.is_available():
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                if 'CUDAExecutionProvider' in available_providers:
                    logger.info("🔍 Testing CUDA provider compatibility...")
                    
                    try:
                        # 配置CUDA提供程序选项
                        cuda_options = {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }
                        
                        # 添加CUDA提供程序
                        providers.append(('CUDAExecutionProvider', cuda_options))
                        logger.info("✅ CUDA provider configured")
                        
                    except Exception as cuda_error:
                        logger.warning(f"⚠️ CUDA provider test failed: {cuda_error}")
                        logger.warning("⚠️ Falling back to CPU provider")
                        
                else:
                    logger.warning("⚠️ CUDA provider not available in onnxruntime")
                    
            except ImportError:
                logger.warning("⚠️ onnxruntime not available")
            except Exception as e:
                logger.warning(f"⚠️ ONNX Runtime setup failed: {e}")
        else:
            logger.warning("⚠️ CUDA not available, using CPU provider")
        
        # 总是添加CPU作为回退
        providers.append('CPUExecutionProvider')
        
        logger.info(f"📝 Final execution providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        return providers
    
    def init_enhanced_face_analyser(self) -> Optional[Any]:
        """初始化增强的人脸分析器"""
        if self.face_analyser is not None:
            return self.face_analyser
            
        try:
            import insightface
            
            # 尝试使用更高精度的buffalo_sc模型
            model_paths = [
                "/runpod-volume/faceswap/buffalo_sc",  # 高精度模型
                "/runpod-volume/faceswap/buffalo_l",   # 标准模型
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        model_name = os.path.basename(model_path)
                        logger.info(f"🔍 Trying face analysis model: {model_name}")
                        
                        self.face_analyser = insightface.app.FaceAnalysis(
                            name=model_name,
                            root=os.path.dirname(model_path),
                            providers=self.execution_providers
                        )
                        
                        # 使用更高分辨率进行检测
                        det_size = (1280, 1280) if model_name == 'buffalo_sc' else (1024, 1024)
                        self.face_analyser.prepare(ctx_id=0, det_size=det_size)
                        
                        logger.info(f"✅ Enhanced face analyser initialized with {model_name} (resolution: {det_size})")
                        return self.face_analyser
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                        continue
            
            logger.error("❌ No compatible face analysis model found")
            return None
            
        except ImportError:
            logger.error("❌ InsightFace not available")
            return None
        except Exception as e:
            logger.error(f"❌ Face analyser initialization failed: {e}")
            return None
    
    def calculate_dynamic_blend_ratio(self, source_face: Any, target_face: Any, 
                                    source_image: np.ndarray, target_image: np.ndarray) -> float:
        """根据皮肤色差和光照条件动态计算混合比例"""
        try:
            # 提取人脸区域
            source_bbox = source_face.bbox.astype(int)
            target_bbox = target_face.bbox.astype(int)
            
            source_face_region = source_image[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]]
            target_face_region = target_image[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]]
            
            # 计算平均颜色
            source_mean_color = np.mean(source_face_region, axis=(0, 1))
            target_mean_color = np.mean(target_face_region, axis=(0, 1))
            
            # 计算颜色差异
            color_diff = np.linalg.norm(source_mean_color - target_mean_color)
            
            # 计算光照差异
            source_brightness = np.mean(cv2.cvtColor(source_face_region, cv2.COLOR_BGR2GRAY))
            target_brightness = np.mean(cv2.cvtColor(target_face_region, cv2.COLOR_BGR2GRAY))
            brightness_diff = abs(source_brightness - target_brightness)
            
            # 动态调整混合比例
            base_ratio = 0.85
            
            # 颜色差异越大，混合比例越低
            color_adjustment = min(color_diff / 100.0, 0.15)
            
            # 光照差异越大，混合比例越低
            brightness_adjustment = min(brightness_diff / 50.0, 0.1)
            
            dynamic_ratio = base_ratio - color_adjustment - brightness_adjustment
            dynamic_ratio = max(0.6, min(0.95, dynamic_ratio))
            
            logger.info(f"🎨 Dynamic blend ratio: {dynamic_ratio:.3f}")
            return dynamic_ratio
            
        except Exception as e:
            logger.warning(f"⚠️ Dynamic blend calculation failed: {e}")
            return 0.85
    
    def init_face_swapper(self) -> Optional[Any]:
        """初始化换脸模型"""
        if self.face_swapper is not None:
            return self.face_swapper
            
        try:
            import insightface
            
            model_path = "/runpod-volume/faceswap/inswapper_128_fp16.onnx"
            if not os.path.exists(model_path):
                logger.error(f"❌ Face swap model not found: {model_path}")
                return None
            
            self.face_swapper = insightface.model_zoo.get_model(
                model_path,
                providers=self.execution_providers
            )
            
            logger.info("✅ Face swapper initialized successfully")
            return self.face_swapper
            
        except Exception as e:
            logger.error(f"❌ Face swapper initialization failed: {e}")
            return None
    
    def init_face_enhancer(self) -> Optional[Any]:
        """初始化脸部增强器"""
        if self.face_enhancer is not None:
            return self.face_enhancer
            
        try:
            from gfpgan import GFPGANer
            
            model_path = "/runpod-volume/faceswap/GFPGANv1.4.pth"
            if not os.path.exists(model_path):
                logger.error(f"❌ GFPGAN model not found: {model_path}")
                return None
            
            self.face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            logger.info("✅ Face enhancer initialized successfully")
            return self.face_enhancer
            
        except Exception as e:
            logger.error(f"❌ Face enhancer initialization failed: {e}")
            return None
    
    def detect_faces_multi_scale(self, image: np.ndarray) -> List[Any]:
        """多尺度人脸检测，提高检测精度"""
        if self.face_analyser is None:
            self.init_enhanced_face_analyser()
            
        if self.face_analyser is None:
            return []
        
        all_faces = []
        
        # 多尺度检测
        scales = [1.0, 1.2, 0.8]  # 原始尺寸、放大、缩小
        
        for scale in scales:
            try:
                if scale != 1.0:
                    h, w = image.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_image = cv2.resize(image, (new_w, new_h))
                else:
                    scaled_image = image
                
                faces = self.face_analyser.get(scaled_image)
                
                # 如果是缩放图像，需要调整坐标
                if scale != 1.0:
                    for face in faces:
                        face.bbox = face.bbox / scale
                        if hasattr(face, 'kps'):
                            face.kps = face.kps / scale
                
                all_faces.extend(faces)
                
            except Exception as e:
                logger.warning(f"⚠️ Multi-scale detection failed at scale {scale}: {e}")
                continue
        
        # 去重：移除重叠的检测结果
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        logger.info(f"🔍 Multi-scale detection found {len(unique_faces)} unique faces")
        return unique_faces
    
    def _remove_duplicate_faces(self, faces: List[Any], iou_threshold: float = 0.5) -> List[Any]:
        """移除重复的人脸检测结果"""
        if len(faces) <= 1:
            return faces
        
        # 计算所有人脸对的IoU
        unique_faces = []
        
        for i, face in enumerate(faces):
            is_duplicate = False
            
            for unique_face in unique_faces:
                iou = self._calculate_bbox_iou(face.bbox, unique_face.bbox)
                if iou > iou_threshold:
                    # 保留置信度更高的
                    if face.det_score > unique_face.det_score:
                        unique_faces.remove(unique_face)
                        unique_faces.append(face)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def apply_region_aware_blending(self, swapped_face: np.ndarray, original_face: np.ndarray, 
                                  face_landmarks: np.ndarray) -> np.ndarray:
        """区域感知融合：对不同面部区域使用不同的融合参数"""
        try:
            # 定义面部区域
            regions = {
                'forehead': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],  # 额头区域
                'cheeks': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 脸颊轮廓
                'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],  # 鼻子
                'eyes': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],  # 眼部
                'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]  # 嘴部
            }
            
            # 不同区域的混合权重
            region_weights = {
                'forehead': 0.7,   # 额头保留更多原始特征
                'cheeks': 0.85,    # 脸颊中等混合
                'nose': 0.9,       # 鼻子更多换脸特征
                'eyes': 0.95,      # 眼部最多换脸特征
                'mouth': 0.9       # 嘴部更多换脸特征
            }
            
            result = swapped_face.copy()
            h, w = result.shape[:2]
            
            for region_name, landmark_indices in regions.items():
                if len(landmark_indices) == 0:
                    continue
                    
                try:
                    # 获取区域关键点
                    region_points = face_landmarks[landmark_indices]
                    
                    # 创建区域掩码
                    mask = np.zeros((h, w), dtype=np.uint8)
                    hull = cv2.convexHull(region_points.astype(np.int32))
                    cv2.fillPoly(mask, [hull], 255)
                    
                    # 应用高斯模糊使边界平滑
                    mask = cv2.GaussianBlur(mask, (15, 15), 0)
                    mask = mask.astype(np.float32) / 255.0
                    
                    # 获取区域权重
                    weight = region_weights.get(region_name, 0.85)
                    
                    # 区域混合
                    for c in range(3):
                        result[:, :, c] = (
                            swapped_face[:, :, c] * mask * weight +
                            original_face[:, :, c] * mask * (1 - weight) +
                            result[:, :, c] * (1 - mask)
                        )
                        
                except Exception as e:
                    logger.warning(f"⚠️ Region blending failed for {region_name}: {e}")
                    continue
            
            logger.info("✨ Applied region-aware blending")
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"⚠️ Region-aware blending failed: {e}")
            return swapped_face
    
    def apply_lighting_correction(self, source_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        """光照匹配：使用色调映射对齐源脸与目标图像光照"""
        try:
            # 转换为LAB色彩空间进行光照分析
            source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
            
            # 分离亮度通道
            source_l = source_lab[:, :, 0]
            target_l = target_lab[:, :, 0]
            
            # 计算亮度统计
            source_mean = np.mean(source_l)
            target_mean = np.mean(target_l)
            source_std = np.std(source_l)
            target_std = np.std(target_l)
            
            # 亮度匹配
            if source_std > 0:
                adjusted_l = (source_l - source_mean) * (target_std / source_std) + target_mean
                adjusted_l = np.clip(adjusted_l, 0, 255)
                
                # 重新组合LAB图像
                adjusted_lab = source_lab.copy()
                adjusted_lab[:, :, 0] = adjusted_l
                
                # 转换回BGR
                result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
                
                logger.info("✨ Applied lighting correction")
                return result
            else:
                return source_image
                
        except Exception as e:
            logger.warning(f"⚠️ Lighting correction failed: {e}")
            return source_image
    
    def enhance_face_with_adaptive_strength(self, face_image: np.ndarray, 
                                          quality_score: float = 0.5) -> Tuple[np.ndarray, bool]:
        """自适应强度的脸部增强"""
        if self.face_enhancer is None:
            self.init_face_enhancer()
            
        if self.face_enhancer is None:
            return face_image, False
        
        try:
            # 根据质量分数调整增强强度
            if quality_score > 0.8:
                strength = 0.5  # 高质量图像使用较低强度
            elif quality_score > 0.6:
                strength = 0.7  # 中等质量使用中等强度
            else:
                strength = 0.9  # 低质量图像使用高强度
            
            # 应用GFPGAN增强
            _, _, enhanced_face = self.face_enhancer.enhance(
                face_image, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True,
                weight=strength
            )
            
            if enhanced_face is not None:
                # 智能混合：保留一些原始细节
                blend_factor = 0.8 if quality_score > 0.7 else 0.9
                final_result = cv2.addWeighted(
                    enhanced_face, blend_factor,
                    face_image, 1 - blend_factor,
                    0
                )
                
                logger.info(f"✅ Enhanced face with adaptive strength: {strength:.2f}")
                return final_result, True
            else:
                return face_image, False
                
        except Exception as e:
            logger.warning(f"⚠️ Face enhancement failed: {e}")
            return face_image, False
    
    def advanced_face_swap(self, source_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """高级换脸处理流程"""
        try:
            logger.info("🔄 Starting advanced face swap pipeline...")
            
            # 1. 多尺度人脸检测
            logger.info("🔍 Multi-scale face detection...")
            source_faces = self.detect_faces_multi_scale(source_image)
            target_faces = self.detect_faces_multi_scale(target_image)
            
            if not source_faces:
                logger.warning("⚠️ No faces detected in source image")
                return target_image, False
                
            if not target_faces:
                logger.warning("⚠️ No faces detected in target image")
                return target_image, False
            
            # 2. 选择最佳人脸（最大面积 + 最高置信度）
            def get_face_score(face):
                bbox = face.bbox
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                return area * face.det_score
            
            source_face = max(source_faces, key=get_face_score)
            target_face = max(target_faces, key=get_face_score)
            
            logger.info(f"✅ Selected source face (confidence: {source_face.det_score:.3f}, area: {get_face_score(source_face):.0f})")
            logger.info(f"✅ Selected target face (confidence: {target_face.det_score:.3f}, area: {get_face_score(target_face):.0f})")
            
            # 3. 光照校正
            logger.info("🌟 Applying lighting correction...")
            source_bbox = source_face.bbox.astype(int)
            source_face_region = source_image[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]]
            corrected_source = self.apply_lighting_correction(source_face_region, target_image)
            
            # 4. 执行换脸
            if self.face_swapper is None:
                self.init_face_swapper()
                
            if self.face_swapper is None:
                logger.error("❌ Face swapper not available")
                return target_image, False
            
            logger.info("🔄 Performing face swap...")
            swapped_result = self.face_swapper.get(target_image, target_face, source_face, paste_back=True)
            
            if swapped_result is None:
                logger.warning("⚠️ Face swap failed")
                return target_image, False
            
            # 5. 动态混合
            logger.info("🎨 Applying dynamic blending...")
            blend_ratio = self.calculate_dynamic_blend_ratio(source_face, target_face, source_image, target_image)
            
            # 创建渐变掩码
            mask = self._create_gradient_mask(target_face, target_image.shape[:2])
            
            # 应用动态混合
            for c in range(3):
                swapped_result[:, :, c] = (
                    swapped_result[:, :, c] * mask * blend_ratio +
                    target_image[:, :, c] * (1 - mask * blend_ratio)
                )
            
            # 6. 区域感知融合（如果有关键点）
            if hasattr(target_face, 'kps') and target_face.kps is not None:
                logger.info("✨ Applying region-aware blending...")
                swapped_result = self.apply_region_aware_blending(
                    swapped_result, target_image, target_face.kps
                )
            
            # 7. 高级后处理
            logger.info("🔧 Applying advanced post-processing...")
            
            # 双边滤波减少伪影
            swapped_result = cv2.bilateralFilter(swapped_result, 9, 80, 80)
            
            # 锐化增强细节
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(swapped_result, -1, kernel)
            swapped_result = cv2.addWeighted(swapped_result, 0.8, sharpened, 0.2, 0)
            
            # 8. 自适应脸部增强
            logger.info("🔧 Applying adaptive face enhancement...")
            quality_score = min(source_face.det_score, target_face.det_score)
            enhanced_result, enhancement_success = self.enhance_face_with_adaptive_strength(
                swapped_result, quality_score
            )
            
            if enhancement_success:
                swapped_result = enhanced_result
            
            logger.info("✅ Advanced face swap completed successfully")
            return swapped_result, True
            
        except Exception as e:
            logger.error(f"❌ Advanced face swap failed: {e}")
            import traceback
            logger.error(f"📝 Traceback: {traceback.format_exc()}")
            return target_image, False
    
    def _create_gradient_mask(self, face, image_shape: Tuple[int, int]) -> np.ndarray:
        """创建渐变掩码用于自然边界融合"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 扩展边界框
        margin = int(min(x2 - x1, y2 - y1) * 0.1)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # 创建椭圆掩码
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        radius_x, radius_y = (x2 - x1) // 2, (y2 - y1) // 2
        
        y_coords, x_coords = np.ogrid[:h, :w]
        ellipse_mask = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2
        
        # 创建渐变
        mask = np.where(ellipse_mask <= 1, 1.0, 0.0)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        return mask

# 全局优化器实例
_face_swap_optimizer = None

def get_face_swap_optimizer() -> AdvancedFaceSwapOptimizer:
    """获取全局换脸优化器实例"""
    global _face_swap_optimizer
    if _face_swap_optimizer is None:
        _face_swap_optimizer = AdvancedFaceSwapOptimizer()
    return _face_swap_optimizer

def process_advanced_face_swap(source_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """处理高级换脸的主要接口"""
    optimizer = get_face_swap_optimizer()
    return optimizer.advanced_face_swap(source_image, target_image) 