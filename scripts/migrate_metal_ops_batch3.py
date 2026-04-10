#!/usr/bin/env python3
"""
migrate_metal_ops_batch3.py
============================
Автоматически конвертирует 26 оставшихся legacy full-kernel MSL файлов
в чистый MLX 0.31.1+ body-only стиль.

Что исправляет:
  1. Удаляет `#include <metal_stdlib>` и `using namespace metal;`
  2. Удаляет полную сигнатуру `[[kernel]] void name(` ... `) {` из source строки
  3. Заменяет `ensure_contiguous=True` → `ensure_row_contiguous=True`
  4. Удаляет `template_type=...` из вызова kernel()
  5. Удаляет `header="..."` из metal_kernel()
  6. Меняет `stream=mx.cpu` → `stream=mx.gpu` где применимо
"""

import re
import sys
from pathlib import Path

MODULES_ROOT = Path(__file__).parent / "modules" / "Metal"

LEGACY_MODULES = [
    "lookahead-mlx/src/lookahead_mlx/infra/ops/lookahead.py",
    "medusa-mlx/src/medusa_mlx/infra/ops/medusa.py",
    "metal-scheduler-mlx/src/metal_scheduler_mlx/infra/ops/scheduler.py",
    "minference-mlx/src/minference_mlx/infra/ops/minfer.py",
    "mla-mlx/src/mla_mlx/infra/ops/mla.py",
    "multi-agent-orchestrator-mlx/src/multi_agent_orchestrator_mlx/infra/ops/router.py",
    "mxfp4-mlx/src/mxfp4_mlx/infra/ops/mxfp4.py",
    "omniquant-mlx/src/omniquant_mlx/infra/ops/omni.py",
    "plan-and-explore-mlx/src/plan_and_explore_mlx/infra/ops/plan.py",
    "polarquant-mlx/src/polarquant_mlx/infra/ops/polar.py",
    "pyramid-kv-mlx/src/pyramid_kv_mlx/infra/ops/pyramid.py",
    "qaq-mlx/src/qaq_mlx/infra/ops/qaq.py",
    "qjl-quant-mlx/src/qjl_quant_mlx/infra/ops/qjl.py",
    "quarot-mlx/src/quarot_mlx/infra/ops/quarot.py",
    "radix-cache-mlx/src/radix_cache_mlx/infra/ops/radix.py",
    "search-decode-mlx/src/search_decode_mlx/infra/ops/search.py",
    "self-spec-mlx/src/self_spec_mlx/infra/ops/selfspec.py",
    "speculative-decode-mlx/src/speculative_decode_mlx/infra/ops/speculative.py",
    "spinquant-mlx/src/spinquant_mlx/infra/ops/spin.py",
    "spotlight-attention-mlx/src/spotlight_attention_mlx/infra/ops/spotlight.py",
    "super-weight-mlx/src/super_weight_mlx/infra/ops/superweight.py",
    "tiered-cache-mlx/src/tiered_cache_mlx/infra/ops/tiered.py",
    "titans-mlx/src/titans_mlx/infra/ops/titans.py",
    "tome-mlx/src/tome_mlx/infra/ops/tome.py",
    "triton-bridge-mlx/src/triton_bridge_mlx/infra/ops/triton_bridge.py",
    "verifiable-rewards-mlx/src/verifiable_rewards_mlx/infra/ops/rewards.py",
]


def extract_body_from_full_kernel(source_block: str) -> str:
    """
    Из полной kernel декларации извлекает только тело функции.
    
    Вход:  #include <metal_stdlib>
           using namespace metal;
           // comment
           [[kernel]] void my_kernel(
               device float* x [[buffer(0)]],
               uint3 gid [[thread_position_in_grid]]
           ) {
               uint i = gid.x;
               out[i] = x[i];
           }
    
    Выход:     uint i = gid.x;
               out[i] = x[i];
    """
    lines = source_block.split('\n')
    result_lines = []
    
    in_signature = False    # внутри [[kernel]] void ... )  {
    in_body = False         # внутри { тела функции }
    brace_depth = 0
    skip_lines = {'#include <metal_stdlib>', 'using namespace metal;'}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Пропускаем #include и using namespace
        if stripped in skip_lines or stripped.startswith('#include'):
            i += 1
            continue
        
        # Начало сигнатуры [[kernel]] void
        if '[[kernel]]' in stripped and 'void' in stripped:
            in_signature = True
            i += 1
            continue
        
        # Пока в сигнатуре — ищем открывающую { и начало тела
        if in_signature:
            if '{' in stripped:
                in_signature = False
                in_body = True
                brace_depth = stripped.count('{') - stripped.count('}')
                # Если после { есть код на той же строке — берём его
                after_brace = stripped[stripped.index('{') + 1:].strip()
                if after_brace and after_brace != '}':
                    result_lines.append('    ' + after_brace)
                if brace_depth <= 0:
                    in_body = False
            i += 1
            continue
        
        # В теле функции — копируем содержимое
        if in_body:
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth <= 0:
                # Закрывающая } тела — конец
                in_body = False
                # Если перед } есть код — добавляем его
                before_close = stripped[:stripped.rfind('}')].strip()
                if before_close:
                    result_lines.append('    ' + before_close)
            else:
                result_lines.append(line)
            i += 1
            continue
        
        # До начала [[kernel]] — это комментарии, сохраняем
        if not in_signature and not in_body and stripped:
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)


def fix_source_string(content: str) -> str:
    """Извлекает тело из MSL_SOURCE = \"\"\"...\"\"\""""
    
    # Паттерн: находим тройные кавычки с MSL кодом
    pattern = re.compile(
        r'(MSL_SOURCE\s*=\s*""")(.*?)(""")',
        re.DOTALL
    )
    
    def replace_source(m):
        prefix = m.group(1)
        body = m.group(2)
        suffix = m.group(3)
        
        # Проверяем — есть ли legacy full-kernel
        if '#include <metal_stdlib>' not in body and '[[kernel]]' not in body:
            return m.group(0)  # уже чистый — не трогаем
        
        # Извлекаем тело
        new_body = extract_body_from_full_kernel(body)
        
        # Убеждаемся что есть отступ и переносы
        if new_body.strip():
            return prefix + '\n' + new_body + '\n' + suffix
        return prefix + '\n' + new_body + '\n' + suffix
    
    return pattern.sub(replace_source, content)


def fix_metal_kernel_call(content: str) -> str:
    """Фиксит параметры в mx.fast.metal_kernel(...)"""
    
    # ensure_contiguous=True → ensure_row_contiguous=True
    # Только если это не уже ensure_row_contiguous
    content = re.sub(
        r'\bensure_contiguous\s*=\s*True',
        'ensure_row_contiguous=True',
        content
    )
    
    # Удаляем header="..." или header='' (любые кавычки)
    content = re.sub(
        r',?\s*header\s*=\s*["\'][^"\']*["\']',
        '',
        content
    )
    
    return content


def fix_kernel_call(content: str) -> str:
    """Фиксит параметры в вызове kernel(...)"""
    
    # Удаляем template_type=... (любое значение до запятой или скобки)
    content = re.sub(
        r',?\s*template_type\s*=\s*[^,\n)]+',
        '',
        content
    )
    
    # stream=mx.cpu → stream=mx.gpu
    content = re.sub(
        r'\bstream\s*=\s*mx\.cpu\b',
        'stream=mx.gpu',
        content
    )
    
    return content


def migrate_file(rel_path: str, dry_run: bool = False) -> dict:
    """Мигрирует один файл. Возвращает статус."""
    full_path = MODULES_ROOT / rel_path
    
    if not full_path.exists():
        return {"file": rel_path, "status": "MISSING", "changes": []}
    
    original = full_path.read_text(encoding='utf-8')
    content = original
    changes = []
    
    # 1. Чистим MSL_SOURCE — удаляем full-kernel декларацию
    if '#include <metal_stdlib>' in content:
        content = fix_source_string(content)
        changes.append("extracted body-only from [[kernel]] void declaration")
    
    # 2. Фиксим параметры metal_kernel()
    if 'ensure_contiguous=True' in content or 'header=' in content:
        content = fix_metal_kernel_call(content)
        changes.append("fixed metal_kernel() params (ensure_row_contiguous, removed header)")
    
    # 3. Фиксим вызов kernel()
    if 'template_type=' in content or 'stream=mx.cpu' in content:
        content = fix_kernel_call(content)
        changes.append("fixed kernel() call (removed template_type, cpu→gpu)")
    
    if content == original:
        return {"file": rel_path, "status": "NO_CHANGES", "changes": []}
    
    if not dry_run:
        full_path.write_text(content, encoding='utf-8')
    
    return {"file": rel_path, "status": "FIXED", "changes": changes}


def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print(f"🔧 migrate_metal_ops_batch3.py")
    print(f"   Mode: {'DRY RUN' if dry_run else 'WRITE'}")
    print(f"   Files: {len(LEGACY_MODULES)}")
    print(f"   Target: MLX 0.31.1+ body-only API")
    print("-" * 60)
    
    results = {"FIXED": [], "NO_CHANGES": [], "MISSING": [], "ERROR": []}
    
    for rel_path in LEGACY_MODULES:
        try:
            result = migrate_file(rel_path, dry_run=dry_run)
            status = result["status"]
            results[status].append(rel_path)
            
            icon = {"FIXED": "✅", "NO_CHANGES": "⏭️", "MISSING": "❌", "ERROR": "💥"}.get(status, "?")
            module_name = rel_path.split('/')[0]
            
            print(f"{icon} {module_name:<45} {status}")
            if verbose and result["changes"]:
                for ch in result["changes"]:
                    print(f"     → {ch}")
        
        except Exception as e:
            results["ERROR"].append(rel_path)
            module_name = rel_path.split('/')[0]
            print(f"💥 {module_name:<45} ERROR: {e}")
    
    print("-" * 60)
    print(f"✅ FIXED:      {len(results['FIXED'])}")
    print(f"⏭️  NO_CHANGES: {len(results['NO_CHANGES'])}")
    print(f"❌ MISSING:    {len(results['MISSING'])}")
    print(f"💥 ERROR:      {len(results['ERROR'])}")
    print()
    
    if not dry_run and results["FIXED"]:
        print(f"🚀 {len(results['FIXED'])} файлов обновлено. Запускай тесты:")
        print(f"   python test_metal_batch3.py")
    elif dry_run:
        print("ℹ️  Dry-run завершён. Запусти без --dry-run для применения изменений.")
    
    return 0 if not results["ERROR"] else 1


if __name__ == "__main__":
    sys.exit(main())
