"""
统一图表样式配置模块
Unified chart style configuration module
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional


# 层级颜色（T0-T3）
TIER_COLORS = {
    0: '#E74C3C',   # 总装厂 - 红
    1: '#3498DB',   # 集成商 - 蓝
    2: '#2ECC71',   # 零部件 - 绿
    3: '#F39C12',   # 原材料 - 橙
}

# 风险等级颜色
RISK_COLORS = {
    'high':   '#E74C3C',   # 红
    'medium': '#F39C12',   # 橙
    'low':    '#F1C40F',   # 黄
    'safe':   '#2ECC71',   # 绿
}

# 层级中文/英文名称
TIER_NAMES = {
    0: ('总装厂', 'OEM Assembly'),
    1: ('分系统集成商', 'Tier-1 Integrator'),
    2: ('零部件供应商', 'Tier-2 Parts Supplier'),
    3: ('原材料供应商', 'Tier-3 Raw Material'),
}


def _find_chinese_font() -> Optional[str]:
    """查找系统中可用的中文字体。"""
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong',
        'KaiTi', 'STSong', 'STHeiti', 'Arial Unicode MS',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return None


def apply_research_style() -> None:
    """应用统一的科研图表样式。

    Sets matplotlib rcParams for publication-quality figures with
    Chinese/English bilingual support.
    """
    chinese_font = _find_chinese_font()

    style = {
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }

    if chinese_font:
        style['font.family'] = 'sans-serif'
        style['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
        style['axes.unicode_minus'] = False

    plt.rcParams.update(style)


def save_figure(fig: plt.Figure, filename: str, output_dir: str = 'outputs/figures') -> None:
    """保存图表为 PDF 和 PNG 两种格式。

    Args:
        fig: matplotlib Figure 对象
        filename: 文件名（不含扩展名），如 'F1-1_network_topology'
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    png_path = os.path.join(output_dir, f'{filename}.png')

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f'  已保存: {pdf_path}')
    print(f'  已保存: {png_path}')
