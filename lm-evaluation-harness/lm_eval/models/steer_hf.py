from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Optional

# 引入 steering_vectors 库
try:
    from steering_vectors import SteeringVector
except ImportError:
    raise ImportError("Please install steering-vectors: pip install steering-vectors")

# lm-eval registry
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

@register_model("steer_hf")
class SteerHFLM(HFLM):
    def __init__(self, **kwargs):
        raw_ma = kwargs.get("model_args", {}) or {}
        ma = self._parse_model_args(raw_ma)
        # 消费自定义参数
        def _consume(key, caster=None, default=None):
            if key in kwargs: val = kwargs.pop(key)
            elif key in ma: val = ma.pop(key)
            else: val = default
            return caster(val) if (caster and val is not None) else val

        self.steer_lambda = _consume("steer_lambda", float, 0.0)
        self.steer_vec_path = _consume("steer_vec_path", str, None)
        self.steer_min_token = _consume("steer_min_token", int, 0)
        
        # ★ 关键参数：指定只干预哪一层
        self.steer_layer = _consume("steer_layer", int, None)

        super().__init__(**kwargs)

        self.steering_vector = None
        self._enabled = (self.steer_lambda != 0.0 and self.steer_vec_path is not None)

        if self._enabled:
            print(f"[SteerHFLM] Loading vector from: {self.steer_vec_path}")
            try:
                # 1. 加载完整对象 (可能包含多个层)
                full_vector = torch.load(self.steer_vec_path, map_location="cpu")
                
                # 兼容性检查：确保加载的是 SteeringVector 对象
                if isinstance(full_vector, dict) and not isinstance(full_vector, SteeringVector):
                    full_vector = SteeringVector(layer_activations=full_vector)

                # 2. ★ 核心修改：如果用户指定了 steer_layer，则进行过滤
                if self.steer_layer is not None:
                    # 检查该层是否存在于文件中
                    if self.steer_layer not in full_vector.layer_activations:
                        available_layers = list(full_vector.layer_activations.keys())
                        raise ValueError(f"Requested steer_layer={self.steer_layer} not found in vector file. Available: {available_layers}")
                    
                    print(f"[SteerHFLM] Filtering vector: keeping ONLY layer {self.steer_layer}")
                    
                    # 创建一个新的 SteeringVector，只包含这一层的数据
                    # layer_activations 是一个 dict {layer_id: tensor}
                    single_layer_data = {self.steer_layer: full_vector.layer_activations[self.steer_layer]}
                    self.steering_vector = SteeringVector(layer_activations=single_layer_data)
                
                else:
                    # 如果没指定层，就默认应用文件里所有的层
                    self.steering_vector = full_vector
                    print(f"[SteerHFLM] No specific layer requested. Applying to ALL layers found: {list(self.steering_vector.layer_activations.keys())}")

                # 3. 移动到正确的设备 (GPU)
                self.steering_vector = self.steering_vector.to(
                    device=self.model.device, 
                    dtype=self.model.dtype
                )

            except Exception as e:
                raise ValueError(f"Failed to load or slice steering vector: {e}")
            
    def _parse_model_args(self, ma):
        """Standard lm-eval arg parser"""
        if isinstance(ma, dict):
            return dict(ma)
        if isinstance(ma, str):
            out = {}
            s = ma.strip()
            if not s: return out
            parts = [p for p in s.split(",") if p.strip() != ""]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    out[k.strip()] = self._coerce(v.strip())
                else:
                    out[p.strip()] = True
            return out
        return {}

    def _coerce(self, v: str):
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]
        vl = v.lower()
        if vl == "true": return True
        if vl == "false": return False
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except:
            return v

    @contextmanager
    def _steering_ctx(self):
        """
        核心修改：使用 steering_vector.apply() 上下文管理器
        """
        if self._enabled and self.steering_vector is not None:
            # apply 会自动处理 hook 注册、device 移动、以及在 forward pass 中注入向量
            # multiplier 即你的 steer_lambda
            with self.steering_vector.apply(self.model, multiplier=self.steer_lambda):
                yield
        else:
            yield

    # ---- 覆盖 lm-eval 的核心生成/评估方法 ----

    def generate_until(self, requests):
        with self._steering_ctx():
            return super().generate_until(requests)

    def loglikelihood(self, requests):
        with self._steering_ctx():
            return super().loglikelihood(requests)

    def loglikelihood_rolling(self, requests):
        with self._steering_ctx():
            return super().loglikelihood_rolling(requests)